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
import dnnlib
from torch_utils import distributed as dist
import matplotlib.pyplot as plt
from moving_mnist import MovingMNIST  # Assuming you have a MovingMNIST loader
from torch.utils.data import DataLoader
from torch_utils.utility import convert_video2images_in_batch
from torch.utils.data import RandomSampler
import math

def plot_curve_of_t_steps(sigma_min, sigma_max, rho, num_steps, net, latents):
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # plot t steps
    plt.plot(t_steps.cpu().numpy())
    plt.xlabel('Step')
    plt.ylabel('t step')
    plt.title('Time step discretization')
    # highest noise level is sigma_max
    plt.axhline(y=sigma_max, color='r', linestyle='--', label=f'sigma_max {sigma_max}')
    # lowest noise level is sigma_min
    plt.axhline(y=sigma_min, color='g', linestyle='--', label=f'sigma_min {sigma_min}')
    plt.legend()
    plt.show()
    plt.close()

def cosine_annealing(epoch, total_epochs, min_lr, max_lr):
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))

def plot_gamma(gamma_arr, S_churn):
    # plot gamma values
    plt.plot(gamma_arr)
    plt.xlabel('Step')
    plt.ylabel('Gamma')
    plt.title(f'Gamma values with S_churn {S_churn}')
    plt.show()
    plt.close()


def plot_the_gradient_norm(norm_of_gradient_pg, num_steps=18, title=' the Particle Guidance Gradient'):
    plt.close()
    # plot the norm of the gradient of the particle guidance
    # all 22 entries are a new plot line
    num_of_lines = len(norm_of_gradient_pg) // num_steps
    lines = []
    for i in range(num_of_lines):
        lines.append(norm_of_gradient_pg[i::num_of_lines])
    for i, line in enumerate(lines):
        plt.plot(line, label=f'Line {i + 1}')
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Value")
    plt.title("Norm of the" + title)
    plt.show()
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).
def compute_distance_matrix(xs, distance='l2'):
    n = xs.shape[0]
    if distance not in ['l2', 'iou']:
        raise ValueError(f"Invalid distance metric: {distance}")

    if distance == 'l2':
        distance_matrix = torch.cdist(xs.flatten(1), xs.flatten(1), p=2)

    # Compute matrix of Intersection over Union (IoU) distances
    elif distance == 'iou':
        xs_flat = xs.flatten(1)  # Flatten each tensor to 1D

        # Iterate through pairs and calculate IoU for each
        distance_matrix = torch.zeros((n, n), device=xs.device)
        for i in range(n):
            for j in range(i + 1, n):
                intersection = torch.min(xs_flat[i], xs_flat[j]).sum()  # Element-wise minimum
                union = torch.max(xs_flat[i], xs_flat[j]).sum()  # Element-wise maximum
                iou = intersection / union
                distance_matrix[i, j] = 1 - iou  # IoU distance is 1 - IoU
                distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

def compute_particle_guidance_grad(xs, gamma=1, alpha=1, distance='l2'):
    with torch.enable_grad():
        xs = (xs.detach().clone() + 1) * 0.5  # transform from value range [-1,1] to [0,1]
        xs.requires_grad = True
        n = xs.shape[0]

        # Compute matrix of L2 distances
        #  xs.flatten(1).shape = torch.Size([8, 1024])
        distance_matrix = compute_distance_matrix(xs, distance=distance)

        # Only consider upper triangular distance matrix, rest are duplicate entries or distance to self
        triu_indices = torch.triu_indices(n, n, offset=1) # triu_indices = torch.Size([2, 28]) = (n^2 - n) / 2
        distance_list = distance_matrix[triu_indices[0], triu_indices[1]]

        # Normalizing factor
        h_t = distance_list.median() ** 2 / math.log(n)

        # Sum of RBF kernels
        rbf_sum = (-distance_list * gamma / h_t).exp().sum()
        rbf_sum = rbf_sum * alpha
        rbf_sum.backward()

        xs_grad = torch.zeros_like(xs)
        # do gradient by hand
        counter = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    xs_grad[i] += 0.5 * -gamma * (xs[i] - xs[j]) * (-distance_matrix[i, j] * gamma / h_t).exp() / h_t
                    counter += 1
        xs_grad_torch = xs.grad

        return xs.grad

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, local_computer=False,
    plot_diffusion=False, image=None, particle_guidance_factor=0,
    gamma_scheduler=False, alpha_scheduler=False,
    particle_guidance_distance='l2', separate_grad_and_PG=False
):
    # Adjust noise levels based on what's supported by the network.
    if net.num_cond_frames > 0:
        cond_frames = image[:, :net.num_cond_frames, :, :]
        cond_frames = cond_frames.repeat(latents.shape[0], 1, 1, 1)
    else:
        cond_frames = None

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    separate_grad_and_PG = separate_grad_and_PG and particle_guidance_factor > 0

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # cosine annealing
    gamma_schedule = [cosine_annealing(s, num_steps, 0, 1) for s in range(num_steps)] if gamma_scheduler else [1 for s in range(num_steps)]
    alpha_schedule = [cosine_annealing(s, num_steps, 0, 1) for s in range(num_steps)] if alpha_scheduler else [1 for s in range(num_steps)]

    # plot t steps
    intermediate_images = []
    intermediate_denoised = []
    intermediate_denoised_prime = []
    intermediate_direction_cur = []
    particle_guidance_grad_images = []
    intermediate_images_output = []

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0] * 0.1
    gamma_arr = []
    S_churn_ = S_churn
    # make dooble the loop if separate_grad_and_PG is True such that PG and the gradient are added in different steps
    extra_loop = 2 if separate_grad_and_PG else 1
    pg_grad_norm = []
    particle_guidance_grad_norm = []
    d_ODE_norm = []
    pg_grad_iou_norm = []
    pg_grad_l2_norm = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        for j in range(extra_loop):

            x_cur = x_next
            S_churn = S_churn_ if j == 0 else 0
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            gamma_arr.append(gamma)
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # if S_churn is 0, gamma is 0, t_hat is equal to t_cur, x_hat is equal to x_cur

            if net.num_cond_frames > 0:
                x_input = torch.cat([cond_frames, x_hat], dim=1)
            else:
                x_input = x_hat

            # Euler step.
            # plot x_input as latent space
            denoised = net(x_input, t_hat, class_labels, num_cond_frames=net.num_cond_frames).to(torch.float64)
            #d_cur = (x_hat - denoised) / t_hat
            pg_grad = compute_particle_guidance_grad(denoised,gamma=gamma_schedule[i], alpha=alpha_schedule[i], distance=particle_guidance_distance)
            pg_grad_l2 = compute_particle_guidance_grad(denoised, gamma=gamma_schedule[i], alpha=alpha_schedule[i],distance='l2')
            pg_grad_iou = compute_particle_guidance_grad(denoised, gamma=gamma_schedule[i], alpha=alpha_schedule[i],distance='iou')
            pg_grad_iou_norm.extend([torch.norm(pg_grad_iou[ii]) for ii in range(pg_grad_iou.shape[0])])
            pg_grad_l2_norm.extend([torch.norm(pg_grad_l2[ii]) for ii in range(pg_grad_l2.shape[0])])

           # pg_grad_normlized = pg_grad
           # for ii in range(pg_grad.shape[0]):
           #     pg_grad_normlized[ii] = pg_grad[ii] / torch.norm(pg_grad[ii])

            particle_guidance_grad = particle_guidance_factor * t_cur * pg_grad
            particle_guidance_grad_norm.extend([torch.norm(particle_guidance_grad[ii]) for ii in range(particle_guidance_grad.shape[0])])
            if torch.isnan(pg_grad).any() or torch.isinf(pg_grad).any() or (separate_grad_and_PG and j == 0):
                #print('Nan or Inf in pg_grad')
                d_cur = (x_hat - denoised) / t_hat
            elif (separate_grad_and_PG and j == 1):
                d_cur = - particle_guidance_grad
            else:
                d_cur = (x_hat - denoised) / t_hat - particle_guidance_grad
            x_next = x_hat + (t_next - t_hat) * d_cur

            d_ODE =  (x_hat - denoised) / t_hat
            d_ODE_norm.extend([torch.norm(d_ODE[ii]) for ii in range(d_ODE.shape[0])])
            # Apply 2nd order correction.
            if i < num_steps - 1:
                if net.num_cond_frames > 0:
                    x_input = torch.cat([cond_frames, x_hat], dim=1)
                else:
                    x_input = x_hat
                denoised = net(x_input, t_next, class_labels, num_cond_frames=net.num_cond_frames).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                if j == 0:
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Save intermediate images.
        # Convert x_next to an image and store it
        intermediate_image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        intermediate_images.append(intermediate_image[0])
        intermediate_images_output.append(x_next)
        if plot_diffusion:
            # Save the first image for plotting

            denoised_image = (denoised * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            intermediate_denoised.append(denoised_image[0])  # Save the first image for plotting


            direction_cur_image = (d_cur * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            intermediate_direction_cur.append(direction_cur_image[0])  # Save the first image for plotting

            try:
                prime_image = (d_prime * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                intermediate_denoised_prime.append(prime_image[0])  # Save the first image for plotting
            except:
                pass

            particle_guidance_grad_image = (particle_guidance_grad * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            particle_guidance_grad_images.append(particle_guidance_grad_image[0])  # Save the first image for plotting


    if plot_diffusion:
        path_saveing = None
        if local_computer:
            path_saveing = '/Users/juliankleutgens/PycharmProjects/edm-main/out/pg_'
            plot_the_gradient_norm(pg_grad_norm, num_steps=num_steps, title=' Particle Guidance Gradient')
            plot_the_gradient_norm(pg_grad_iou_norm, num_steps=num_steps, title=' Particle Guidance Gradient IoU')
            plot_the_gradient_norm(pg_grad_l2_norm, num_steps=num_steps, title=' Particle Guidance Gradient L2')
            plot_the_gradient_norm(particle_guidance_grad_norm, num_steps=num_steps, title=' Particle Guidance Gradient after sacling  PG')
            #plot_the_gradient_norm(d_ODE_norm, num_steps=num_steps, title='the greadient of ODE')
        plot_diffusion_process(intermediate_denoised, variable_name='Denoised', save_path=path_saveing)
        #plot_diffusion_process_conditional(intermediate_images, images=image, save_path=path_saveing)
        #plot_diffusion_process(intermediate_direction_cur, variable_name='ODE Direction', save_path=path_saveing)
        #plot_diffusion_process(intermediate_denoised_prime, variable_name='Second Order Correction', save_path=path_saveing)
        #plot_diffusion_process(particle_guidance_grad_images, variable_name='Particle Guidance Grad', save_path=path_saveing)
    #plot_gamma(gamma_arr, S_churn)
    return x_next, intermediate_images_output


# Function to plot the intermediate images
def plot_diffusion_process(intermediate_images, num_rows=2, num_cols=5, variable_name='Image', save_path=None):
    if num_rows * num_cols < len(intermediate_images):
        num_rows = (len(intermediate_images) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < len(intermediate_images):
            ax.imshow(intermediate_images[i], cmap='gray')
            ax.set_title(f'Step {i + 1}')
        ax.axis('off')
    title = 'Denoising process: ' + variable_name
    plt.suptitle(title)
    # save images to disk
    if save_path is not None:
        save_path = save_path + f'_{variable_name}.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def plot_diffusion_process_conditional(intermediate_images, num_rows=2, num_cols=5,images=None, save_path=None):
    if images is not None:
        num_frames = images.shape[1]
        images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    else:
        num_frames = 0
    if num_rows * num_cols < len(intermediate_images) + num_frames:
        num_rows = (len(intermediate_images)+num_frames + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i, ax in enumerate(axes.flatten()):
        if i < num_frames:
            ax.imshow(images[0,:,:, i], cmap='gray')
            ax.set_title(f'Frame {i + 1}')
        elif i - num_frames < len(intermediate_images)  and i >= num_frames:
            ax.imshow(intermediate_images[i-num_frames], cmap='gray')
            ax.set_title(f'Step {i- num_frames+ 1}')
        ax.axis('off')
    plt.suptitle('Denoising process: Conditional + Images steps') if num_frames > 0 else plt.suptitle(
        'Denoising process: Images steps')
    if save_path is not None:
        save_path = save_path + '_images_cond.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()


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

def generate_images_and_save_heatmap(
        network_pkl, outdir, num_images=100, max_batch_size=1, num_steps=18,particle_guidance_factor=0,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
):
    """Generate images with S_churn=0.9 and create a heatmap of pixel intensities."""

    # Initialize the MovingMNIST dataset
    dataset_obj = MovingMNIST(train=True, data_root='./data', seq_len=5, num_digits=1, image_size=32,
                              deterministic=False)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=max_batch_size))
    image, labels = next(dataset_iterator)
    image = image.to(device)

    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    images, labels = convert_video2images_in_batch(images=image, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)
    images = images.to(device).to(torch.float32) * 2 - 1
    image = images[:1, :, :, :]

    # Store sum of images for the heatmap
    image_sum = None
    generated_images = []
    s_noise = 1
    for seed in range(num_images):
        latents = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)

        # Generate the image with S_churn=0.9
        generated_img, _ = edm_sampler(
            net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, S_churn=S_churn, image=image, plot_diffusion=False, S_noise=s_noise, particle_guidance_factor=particle_guidance_factor
        )

        # Convert generated image to numpy for processing
        img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        generated_images.append(img_np[0])

        if image_sum is None:
            image_sum = img_np[0].astype(np.float32)
        else:
            image_sum += img_np[0].astype(np.float32)

        # Save the individual generated images
        img_path = os.path.join(outdir, f'generated_image_{seed:03d}.png')
        PIL.Image.fromarray(img_np[0]).save(img_path)

    # Average the pixel intensities to compute the heatmap
    image_mean = image_sum / num_images
    generate_heatmap(image_mean, outdir)


def generate_heatmap(image_mean, outdir):
    """Generate and save a heatmap based on the mean pixel intensities of generated images."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_mean[:, :, 0], cmap='hot', interpolation='nearest')  # Assuming grayscale images
    plt.colorbar(label='Pixel Intensity')
    plt.title('Heatmap of Generated Images')

    heatmap_path = os.path.join(outdir, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.show()



@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

@click.option('--local_computer',        help='Use local computer',                                              is_flag=True)
@click.option('--wandb_run_id',          help='W&B run ID',                                                      type=str, default=None)
@click.option('--particle_guidance_factor', help='Particle guidance factor',                                      type=float, default=0)

def main(network_pkl, outdir, wandb_run_id, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    # Initialize W&B using the provided run ID
    if wandb_run_id:
        import wandb
        wandb.init(id=wandb_run_id, resume="allow")
    if sampler_kwargs["local_computer"]:
        device = torch.device('cpu')

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    particle_guidance_factor = sampler_kwargs["particle_guidance_factor"]



    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dataset_obj = MovingMNIST(train=True, data_root='./data', seq_len=32, num_digits=1, image_size=32, mode='circle',
                              deterministic=False, log_direction_change=True, step_length=0.1, let_last_frame_after_change=False, use_label=True)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    #dataset_sampler = RandomSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=1))
    image_data, labels, direction_change = next(dataset_iterator)
    image_seq = image_data.to(device)
    image_seq1 = image_seq[:,:,:,:,0]
    images, labels = convert_video2images_in_batch(images=image_seq, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)
    from generate_heatmap import calculate_centroids, plot_images_with_centroids
    centroids = calculate_centroids(image=image_seq1.permute(1, 0, 2, 3).to(torch.device('cpu')) )
    plot_images_with_centroids(image=image_seq1.permute(1, 0, 2, 3).to(torch.device('cpu')) , centroids=centroids, local_computer=sampler_kwargs["local_computer"])
    digit = torch.argmax(labels[0, 0, :]).item() + 1
    images = images.to(device).to(torch.float32) * 2 - 1
    idx = direction_change[:,1] - net.num_cond_frames + 1
    image = images[int(idx):int(idx)+1, :, :, :]
    if net.num_cond_frames == 0:
        image = None

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    i = 0
    plot_diffusion = True
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = edm_sampler
        if i == 1:
            plot_diffusion = True
        images, intermediate_images = sampler_fn(net, latents, class_labels,image=image,
                                                 randn_like=rnd.randn_like,plot_diffusion=plot_diffusion, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
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
            #plt.show()
            plt.axis('off')
            # Save the figure as displayed by Matplotlib
            image_path_plt = image_path.replace('.png', '_plt.png')
            plt.savefig(image_path_plt, bbox_inches='tight', pad_inches=0)
            plt.close()
        # Log image to W&B
            if not wandb_run_id == None:
                wandb.log({"Generated Image": wandb.Image(img)})


    # Finish W&B run
    if wandb_run_id:
        wandb.finish()

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
