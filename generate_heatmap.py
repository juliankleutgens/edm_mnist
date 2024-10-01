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
from torch_utils.utility import convert_video2images_in_batch
from generate_conditional_frames import edm_sampler
import tqdm
from torch.utils.data import DataLoader, RandomSampler

def generate_images_and_save_heatmap(
        network_pkl, outdir, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
):
    """Generate images with S_churn=0.9 and create a heatmap of pixel intensities."""

    # Initialize the MovingMNIST dataset
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)


    dataset_obj = MovingMNIST(train=True, data_root='./data', seq_len=32, num_digits=1, image_size=32, move_horizontally=True,
                              deterministic=False, log_direction_change=True, step_length=0.1, let_last_frame_after_change=False)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    dataset_sampler = RandomSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=max_batch_size))
    image, labels, direction_change = next(dataset_iterator)
    polt_images_highlight_direction_change(image, direction_change)
    image_seq = image.to(device)
    image_seq1 = image_seq[:,:,:,:,0]
    images, labels = convert_video2images_in_batch(images=image_seq, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)

    centroids = calculate_centroids(image=image_seq1.to(device) )
    #plot_images_with_centroids(image=image_seq1.to(device) , centroids=centroids)
    images = images.to(device).to(torch.float32) * 2 - 1
    idx = direction_change[:,1] - net.num_cond_frames +1
    image = images[int(idx):int(idx)+1, :, :, :]

    centroids = calculate_centroids(image=(image.to(device) + 1) / 2)
    plot_images_with_centroids(image=(image.to(device) + 1) / 2, centroids=centroids)

    #img_cent = (image * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()



    # Store sum of images for the heatmap
    image_sum = None
    j = 0
    generated_images = []
    with tqdm.tqdm(total=num_images, desc="Generating images") as pbar:
        for seed in range(num_images):
            latents = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            # Generate the image with S_churn=0.9
            generated_img, _ = edm_sampler(
                net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, S_churn=S_churn, image=image, plot_diffusion=False
            )


            img_btw_0_1 = (generated_img + 1) / 2
            # thearshold all values below 0.1 to 0
            img_btw_0_1[img_btw_0_1 < 0.55] = 0
            img_cat_btw_0_1 = torch.cat((img_cat_btw_0_1, img_btw_0_1),dim=1) if 'img_cat_btw_0_1' in locals() else img_btw_0_1

            img_np = (img_btw_0_1 * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            #img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
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
    #plot_images(image)
    generate_heatmap(image_mean, outdir)
    centroids_all = calculate_centroids(image=img_cat_btw_0_1)
    went_to_left = []
    went_to_right = []
    i = 0
    for c in centroids_all:
        if c[0] > centroids[-2][0]:
            went_to_right.append(i)
        else:
            went_to_left.append(i)
        i += 1
    num_times_went_to_right = len(went_to_right)
    print(f'Number of times went to right: {num_times_went_to_right} out of the {num_images} images')
    print(f'This gives us the probability of going to right: {num_times_went_to_right/num_images}')
    print(f"The idx of going left are: {went_to_left}")


    plot_images_with_centroids_reference(image=img_cat_btw_0_1, centroids=centroids_all, centroids_reference=centroids[-2:], indexes=went_to_left)

def generate_heatmap(image_mean, outdir):
    """Generate and save a heatmap based on the mean pixel intensities of generated images."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_mean[:, :, 0], cmap='hot', interpolation='nearest')  # Assuming grayscale images
    plt.colorbar(label='Pixel Intensity')
    plt.title('Heatmap of Generated Images')

    #heatmap_path = os.path.join(outdir, 'heatmap.png')
    #plt.savefig(heatmap_path)
    plt.show()


import numpy as np


def plot_images_with_centroids(image, centroids):
    num_of_images = image.shape[1]
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))

    for i in range(num_of_images):
        ax[i].imshow(image[0, i, :, :], cmap='gray')
        ax[i].scatter(centroids[i][0], centroids[i][1], color='red', marker='x')  # Plot centroids
        # add an arrow to show the direction of the centroid
        if i < num_of_images - 1:
            dx = centroids[i + 1][0] - centroids[i][0]
            dy = centroids[i + 1][1] - centroids[i][1]
            #norm of vector (dx, dy) should be 3
            norm = np.sqrt(dx**2 + dy**2)
            dx /= norm
            dy /= norm
            dx *= 3
            dy *= 3
            ax[i].arrow(centroids[i][0], centroids[i][1], dx, dy, head_width=1, head_length=1, fc='blue', ec='blue')
        #ax[i].scatter(0, 0, color='blue', marker='o')  # Plot centroids
        ax[i].axis('off')

    plt.show()

def plot_images_with_centroids_reference(image, centroids, centroids_reference, indexes=None):
    num_of_images = image.shape[1]
    if num_of_images > 10 and indexes is not None:
        idx_to_plot = indexes[:20] if len(indexes) > 20 else indexes + [i for i in range(10 - len(indexes))]
    elif num_of_images > 10 and indexes is None:
        idx_to_plot = [i for i in range(10)]
    else:
        idx_to_plot = [i for i in range(num_of_images)]

    fig, ax = plt.subplots(1, len(idx_to_plot), figsize=(len(idx_to_plot), 2))

    j = 0
    for i in idx_to_plot:

        ax[j].imshow(image[0, i, :, :], cmap='gray')
        ax[j].scatter(centroids[i][0], centroids[i][1], color='red', marker='x')  # Plot centroids

        # Add an arrow pointing toward the exact ith centroid from the reference point
        dx = centroids[i][0] - centroids_reference[0][0]
        dy = centroids[i][1] - centroids_reference[0][1]

        # Normalize and scale the vector to length 3 if necessary
        norm = np.sqrt(dx ** 2 + dy ** 2)
        dx_scaled = dx / norm * 3
        dy_scaled = dy / norm * 3

        point_vec_x = centroids[i][0] - dx_scaled
        point_vec_y = centroids[i][1] - dy_scaled

        ax[j].arrow(point_vec_x, point_vec_y, dx_scaled, dy_scaled,
                    head_width=1, head_length=1, fc='blue', ec='blue')

        #ax[i].scatter(centroids_reference[0][0], centroids_reference[0][1], color='blue', marker='o')  # Plot reference point
        #ax[i].scatter(centroids_reference[0][0], centroids_reference[1][1], color='green', marker='o')
        ax[j].axis('off')

        #ax[i].scatter(0, 0, color='blue', marker='o')  # Plot centroids
        j += 1

    plt.show()

def calculate_centroids(image):
    # Convert the image tensor to NumPy array (if necessary)
    image_np = image.numpy() if hasattr(image, 'numpy') else image

    # Get the number of images
    num_of_images = image_np.shape[1]

    centroids = []

    for i in range(num_of_images):
        img = image_np[0, i, :, :]
        # Get the coordinates of all pixels
        y, x = np.indices(img.shape)
        # Calculate the weighted sum of pixel positions
        total_intensity = np.sum(img)
        if total_intensity == 0:  # Avoid division by zero
            centroids.append((0, 0))
        else:
            x_centroid = np.sum(x * img) / total_intensity
            y_centroid = np.sum(y * img) / total_intensity
            centroids.append((x_centroid, y_centroid))

    return centroids


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
    for i in range(10):
        generate_images_and_save_heatmap(
            network_pkl=network_pkl, outdir=outdir, num_images=num_images, max_batch_size=max_batch_size,
            num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, S_churn=s_churn, rho=rho, local_computer=local_computer, device=device
        )


if __name__ == "__main__":
    main()
