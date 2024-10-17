import os
import random
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
from torch_utils.utility import convert_video2images_in_batch
from generate_conditional_frames import edm_sampler
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from scipy.stats import binom
import time
from collections import Counter
import wandb
from generate_heatmap import plot_heatmap_of_centroids, plot_images_with_centroids_reference, calculate_centroids, get_directions, get_true_probability, get_direction_mapping
from generate_conditional_frames import edm_sampler

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


# ----------------------------- generating function -----------------------------
# ------------------------------------------------------------------------------
def generate_images_and_save_heatmap_PG(
        network_pkl, outdir, moving_mnist_path, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
        ,mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None, S_noise=1):
    """Generate images with S_churn=0.9 and create a heatmap of pixel intensities."""


    if local_computer:
        device = torch.device('cpu')

    seeds = [i for i in range(num_images)]
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

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

    # ------------------- Initialize the MovingMNIST dataset -------------------
    device_cpu = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # get dataset
    dataset_obj = MovingMNIST(train=True, data_root=moving_mnist_path, seq_len=32, num_digits=1, image_size=32, mode=mode,
                              deterministic=False, log_direction_change=True, step_length=0.1, let_last_frame_after_change=False, use_label=True,
                              num_of_directions_in_circle=num_of_directions,digit_filter=digit_filter)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    #dataset_sampler = RandomSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=1))

    # get the first sequence images
    image_data, labels, direction_change = next(dataset_iterator)
    image_seq = image_data.to(device)
    images, labels = convert_video2images_in_batch(images=image_seq, labels=labels, use_label=False,num_cond_frames=net.num_cond_frames)
    idx = direction_change[:,1] - net.num_cond_frames + 1
    image = images[int(idx):int(idx)+1, :, :, :]

    # get the digit and convert the images to the right format
    digit = torch.argmax(labels[0, 0, :]).item() + 1
    images = images.to(device).to(torch.float32) * 2 - 1

    # calculate the centroids of the first sequence
    centroids = calculate_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2)
    directions = get_directions(num_of_directions, mode)


    image_sum = None
    j = 0
    generated_images = []
    iterator = tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)) if local_computer else rank_batches
    #  ------------------------- for loop to generate the images -------------------------
    for batch_seeds in iterator:
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

        # Generate the images
        generated_img, _ = edm_sampler(
            net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, S_churn=S_churn, image=image, plot_diffusion=True, randn_like=rnd.randn_like,
            S_noise=S_noise, particle_guidance_factor=particle_guidance_factor)

        img_btw_0_1 = (generated_img + 1) / 2
        img_cat_btw_0_1 = torch.cat((img_cat_btw_0_1, img_btw_0_1),dim=0) if 'img_cat_btw_0_1' in locals() else img_btw_0_1
        img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        generated_images.append(img_np[0])


    # calculate the centroids
    centroids_estimated = calculate_centroids(image=img_cat_btw_0_1)

    # calculate the estimated directions
    estimated_directions = []
    i = 0
    for c in centroids_estimated:
        x_t_prev = centroids[-2][0]
        y_t_prev = centroids[-2][1]
        vector = np.array([c[0] - x_t_prev, c[1] - y_t_prev])
        vector_norm = vector / np.linalg.norm(vector)
        directions_norm = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        cosine_similarities = np.dot(directions_norm, vector_norm)
        closest_index = np.argmax(cosine_similarities)
        estimated_directions.append(closest_index)
    count = dict(Counter(estimated_directions))
    prob_estimated = {k: v / num_images for k, v in count.items()}

    if local_computer:
        plot_heatmap_of_centroids(centroids=centroids_estimated)
        plot_images_with_centroids_reference(image=img_cat_btw_0_1, centroids=centroids_estimated, centroids_reference=centroids[-2:])

    return prob_estimated, digit

# ----------------------------- main function -----------------------------
# -------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir', help='Where to save the output images and heatmap', metavar='DIR', type=str, required=True)
@click.option('--max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=8)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1),
              default=22)
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=7)
@click.option('--local_computer', help='Use local computer', is_flag=True)
@click.option('--num_of_directions', help='Number of directions to sample', metavar='INT', type=click.IntRange(min=1),
              default=8)
@click.option('--num_seq', help='Number of sequences', metavar='INT', type=click.IntRange(min=1),
              default=11)
@click.option('--moving_mnist_path', help='Path to the moving mnist dataset', metavar='STR', type=str, required=True)
@click.option('--digit_filter', help='Filter the digit to generate', metavar='INT', type=int, default=None)

def main(network_pkl, outdir, max_batch_size, num_steps, rho,moving_mnist_path,
         local_computer,sigma_min=0.002, sigma_max=80, true_probability=None, num_seq=1, mode='circle', num_of_directions=8, digit_filter=None):
    device = torch.device('cpu' if local_computer else 'cuda')
    results = []
    mean_uniform = []
    dist.init()

    p_true = get_true_probability(true_probability, network_pkl, mode, num_of_directions)
    if mode == 'circle' and num_of_directions == 4 or num_of_directions == 8:
        direction_mapping = get_direction_mapping(num_of_directions)
        stored_probabilities = {key: [] for key in direction_mapping.values()}


    particle_guidance_factor = 0
    wandb.init(project="edm_generation")
    wandb.config.update({"network_pkl": network_pkl, "outdir": outdir, "num_images": 8, "max_batch_size": max_batch_size,
                         "num_steps": num_steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "S_churn": 0, "rho": rho,
                         "local_computer": local_computer, "device": device, "true_probability": p_true, "num_seq": num_seq, "mode":mode,
                         "num_of_directions":num_of_directions, "particle_guidance_factor":particle_guidance_factor})

    digit_prob = {str(i): [] for i in range(10)}
    if digit_filter is not None:
        digit_filter = [digit_filter]

    S_noise_iterater = [0, 0.5, -1, -1.5, -2]
    S_noise_logarithmic = 10 ** np.array(S_noise_iterater)
    particle_guidance_factor_iterater = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    particle_guidance_factor_logarithmic = 10 ** np.array(particle_guidance_factor_iterater)
    num_images = num_of_directions
    s_churn = 0

    for S_noise in S_noise_logarithmic:
        for particle_guidance_factor in particle_guidance_factor_logarithmic:
            prob_estimated, digit = generate_images_and_save_heatmap_PG(
                network_pkl=network_pkl, outdir=outdir, num_images=num_images, max_batch_size=max_batch_size,
                num_steps=num_steps, mode=mode, num_of_directions=num_of_directions,
                sigma_min=sigma_min, sigma_max=sigma_max, S_churn=s_churn, rho=rho, local_computer=local_computer, device=device, moving_mnist_path=moving_mnist_path,
                particle_guidance_factor=particle_guidance_factor, digit_filter=digit_filter, S_noise=S_noise,

            )
            results.append(prob_estimated)


if __name__ == "__main__":
    main()
