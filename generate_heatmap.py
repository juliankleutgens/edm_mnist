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
from generate_conditional_frames import edm_sampler
import tqdm

def generate_images_and_save_heatmap(
        network_pkl, outdir, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
):
    """Generate images with S_churn=0.9 and create a heatmap of pixel intensities."""

    # Initialize the MovingMNIST dataset
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)


    dataset_obj = MovingMNIST(train=True, data_root='./data', seq_len=16, num_digits=1, image_size=32,
                              deterministic=False, direction_change=True, step_length=0.1)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=max_batch_size))
    image, labels, direction_change = next(dataset_iterator)
    polt_images_highlight_direction_change(image, direction_change)
    image = image.to(device)
    images, labels = convert_video2images_in_batch(images=image, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)
    images = images.to(device).to(torch.float32) * 2 - 1
    idx = direction_change[1] - net.num_cond_frames +1
    image = images[idx, :, :, :]



    # Store sum of images for the heatmap
    image_sum = None
    generated_images = []
    with tqdm.tqdm(total=num_images, desc="Generating images") as pbar:
        for seed in range(num_images):
            latents = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            # Generate the image with S_churn=0.9
            generated_img, _ = edm_sampler(
                net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, S_churn=S_churn, image=image, plot_diffusion=False
            )

            # Convert generated image to numpy for processing
            img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            generated_images.append(img_np[0])

            if image_sum is None:
                image_sum = img_np[0].astype(np.float32)
            else:
                image_sum += img_np[0].astype(np.float32)
            pbar.update(1)

            # Save the individual generated images
    #        img_path = os.path.join(outdir, f'generated_image_{seed:03d}.png')
    #        PIL.Image.fromarray(img_np[0]).save(img_path)

    # Average the pixel intensities to compute the heatmap
    image_mean = image_sum / num_images
    plot_images(image)
    generate_heatmap(image_mean, outdir)


def generate_heatmap(image_mean, outdir):
    """Generate and save a heatmap based on the mean pixel intensities of generated images."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_mean[:, :, 0], cmap='hot', interpolation='nearest')  # Assuming grayscale images
    plt.colorbar(label='Pixel Intensity')
    plt.title('Heatmap of Generated Images')

    #heatmap_path = os.path.join(outdir, 'heatmap.png')
    #plt.savefig(heatmap_path)
    plt.show()

def plot_images(image):
    num_of_images = image.shape[1]
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))
    for i in range(num_of_images):
        ax[i].imshow(image[0, i, :, :], cmap='gray')
        ax[i].axis('off')
    plt.show()

def polt_images_highlight_direction_change(image, direction_change):
    from matplotlib.patches import Rectangle
    num_of_images = image.shape[1]
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))
    for i in range(num_of_images):
        ax[i].imshow(image[0, i, :, :], cmap='gray')
        if i in direction_change:
            rect = Rectangle((0, 0), image.shape[2], image.shape[3],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax[i].add_patch(rect)
            ax[i].set_title('Direction Change', fontsize=8)
        ax[i].axis('off')
    plt.show()


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir', help='Where to save the output images and heatmap', metavar='DIR', type=str, required=True)
@click.option('--num_images', help='Number of images to generate', metavar='INT', type=click.IntRange(min=1),
              default=100)
@click.option('--max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=1)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1),
              default=18)
@click.option('--sigma_min', help='Lowest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=0.002)
@click.option('--sigma_max', help='Highest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=80)
@click.option('--s_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0.9,
              show_default=True)
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=7)
@click.option('--local_computer', help='Use local computer', is_flag=True)

def main(network_pkl, outdir, num_images, max_batch_size, num_steps, sigma_min, sigma_max, s_churn, rho,
         local_computer):
    device = torch.device('cpu' if local_computer else 'cuda')
    generate_images_and_save_heatmap(
        network_pkl=network_pkl, outdir=outdir, num_images=num_images, max_batch_size=max_batch_size,
        num_steps=num_steps,
        sigma_min=sigma_min, sigma_max=sigma_max, S_churn=s_churn, rho=rho, local_computer=local_computer, device=device
    )


if __name__ == "__main__":
    main()
