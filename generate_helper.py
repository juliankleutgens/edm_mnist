# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
from copy import deepcopy
import dnnlib
from torch_utils import distributed as dist
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, plot_diffusion=False, local_computer= False, image=None
):
    if net.num_cond_frames > 0:
        cond_frames = image[:, :net.num_cond_frames,:,:]
        # repeat cond_frames to match the batch size of latents
        cond_frames = cond_frames.repeat(latents.shape[0], 1, 1, 1)
    else:
        cond_frames = None
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    intermediate_images = []

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        if net.num_cond_frames > 0:
            x_input = torch.cat([cond_frames, x_hat], dim=1)
        else:
            x_input = x_hat
        denoised = net(x_input, t_hat, class_labels, num_cond_frames=net.num_cond_frames).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            if net.num_cond_frames > 0:
                x_input = torch.cat([cond_frames, x_next], dim=1)
            else:
                x_input = x_hat
            denoised = net(x_input, t_next, class_labels, num_cond_frames=net.num_cond_frames).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Save intermediate images.
        # Convert x_next to an image and store it
        if plot_diffusion:
            intermediate_image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            intermediate_images.append(intermediate_image[0])  # Save the first image for plotting
    return x_next, intermediate_images

def plot_diffusion_process(intermediate_images, num_rows=2, num_cols=5, save_path=None):
    if num_rows * num_cols < len(intermediate_images):
        num_rows = (len(intermediate_images) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < len(intermediate_images):
            ax.imshow(intermediate_images[i], cmap='gray')
            ax.set_title(f'Step {i + 1}')
        ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

def plot_diffusion_process_conditional(intermediate_images, num_rows=2, num_cols=5,images=None, save_path=None):
    num_frames = images.shape[1]
    images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    if num_rows * num_cols  < len(intermediate_images)+ num_frames:
        num_rows = (len(intermediate_images)+num_frames + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i, ax in enumerate(axes.flatten()):
        if i < num_frames:
            ax.imshow(images[0,:,:, i], cmap='gray')
            ax.set_title(f'Frame {i + 1}')
        elif i < len(intermediate_images) + num_frames and i >= num_frames:
            ax.imshow(intermediate_images[i-num_frames], cmap='gray')
            ax.set_title(f'Step {i- num_frames+ 1}')
        ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()
#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, local_computer=False
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
def generate_images_during_training(network_pkl, outdir, seeds, max_batch_size, wandb_run_id=None, device=torch.device('cuda'), local_computer=False,
                                    dist=None, net=None, class_idx=None, subdirs=False, image=None, label=None):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    sampler_kwargs = {
#        "seeds": seeds,  # Adjust seeds as needed
#        "max_batch_size": max_batch_size,
#        "class_idx": class_idx,
        "num_steps": 18,
        "sigma_min": None,  # Use default value
        "sigma_max": None,  # Use default value
        "rho": 7,
        "S_churn": 0,
        "S_min": 0,
        "S_max": float('inf'),
        "S_noise": 1,
        "solver": None,  # Optional
        "discretization": None,  # Optional
        "schedule": None,  # Optional
        "scaling": None,  # Optional
#        "subdirs": outdir,
        "local_computer": local_computer,
#        "wandb_run_id":  wandb_run_id # Replace with actual W&B run ID if needed
    }

    # Resume W&B run if needed
    if wandb_run_id:
        import wandb
        if not wandb.run:
            wandb.init(id=wandb_run_id, resume="allow")

    #if sampler_kwargs["local_computer"]:
    device = torch.device('cpu')


    #dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        print(f'Rank {dist.get_rank()} waiting for rank 0...')
        #torch.distributed.barrier()

    # Load network.
    if net is None:
        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
    else: # Use the provided network
        dist.print0(f'Using provided network...')
        net = deepcopy(net).to(device)
        image = image.to(device)


    # Other ranks follow.
    if dist.get_rank() == 0:
        print(f'Rank 0 is ready, other ranks can start...')
        #torch.distributed.barrier()

    # Loop over batches.
    #dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    print(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        #torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            print(f'Generating images for class index: {class_idx}')
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        plot_diffusion = True
        images, intermediate_images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, plot_diffusion=plot_diffusion, image=image, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        wandb_images = []
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                img = PIL.Image.fromarray(image_np[:, :, 0], 'L')
                plt.imshow(image_np[:, :, 0], cmap='gray')
                img.save(image_path)
            else:
                img = PIL.Image.fromarray(image_np, 'RGB')
                img.save(image_path)
            plt.axis('off')
            # Save the figure as displayed by Matplotlib
            image_path_plt = image_path.replace('.png', '_plt.png')


            plt.savefig(image_path_plt, bbox_inches='tight', pad_inches=0)
            plt.close()

            image_path_steps = image_path.replace('.png', '_steps.png')
            if net.num_cond_frames > 0:
                plot_diffusion_process_conditional(intermediate_images, images=image, save_path=image_path_steps)
            else:
                plot_diffusion_process(intermediate_images, save_path=image_path_steps)
            if wandb.run is not None:
                wandb_image = wandb.Image(image_path_steps, caption=f"Seed: {seed}")
                wandb_images.append(wandb_image)


            # Log all images to W&B
        if wandb.run is not None:
            if class_idx is not None:
                wandb.log({"class_idx": class_idx, "generated_images": wandb_images})
            else:
                wandb.log({"generated_images": wandb_images})
            print(f'Logged {len(wandb_images)} images to W&B')
            print(f'The W&B run URL is: {wandb.run.get_url()}')


    # Done.
    #torch.distributed.barrier()
    dist.print0('Done.')


#----------------------------------------------------------------------------
if __name__ == "__main__":
    outdir = "out/"
    batch = 1
    network = "/Users/juliankleutgens/PycharmProjects/edm-main/cluster_output/00115-8xT4/network-snapshot-002502.pkl"
    local_computer = True
    steps = 18
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sigma_min = 0.002
    sigma_max = 80
    S_noise = 1
    max_batch_size = 1
    dist.init()
    generate_images_during_training(network_pkl=network, outdir=outdir, seeds=seeds, local_computer=local_computer, device=torch.device('cpu'), dist=dist, max_batch_size=max_batch_size)
    print("Done.")