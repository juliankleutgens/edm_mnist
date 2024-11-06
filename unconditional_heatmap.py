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
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from scipy.stats import binom
import time
from collections import Counter
import wandb
from generate_conditional_frames import edm_sampler
from resnet_classifier import get_prediction
from generate_heatmap import edm_sampler2
from trajectory_of_diffusion import add_new_unlabeled_images_into_2d_featuremap


# ----------------------------- utility functions -----------------------------
# -----------------------------------------------------------------------------
def generate_heatmap(image_mean, outdir,local_computer=False):
    """Generate and save a heatmap based on the mean pixel intensities of generated images."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_mean[:, :, 0], cmap='hot', interpolation='nearest')  # Assuming grayscale images
    plt.colorbar(label='Pixel Intensity')
    plt.title('Heatmap of Generated Images')

    # save the image in out folder with a time stamp
    t = time.localtime()
    path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
    plt.savefig(path)
    # save it to wandb
    import wandb
    wandb.log({"generated_images_mean": wandb.Image(path)})
    if local_computer:
        jjj = 0
        plt.show()
    plt.close()

def plot_images_with_centroids(image, centroids, local_computer=False):
    num_of_images = image.shape[0]
    plt.close()
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))

    for i in range(num_of_images):
        ax[i].imshow(image[i, 0, :, :], cmap='gray')
        ax[i].scatter(centroids[i][0], centroids[i][1], color='red', marker='x')  # Plot centroids
        # add an arrow to show the direction of the centroid
        if i < num_of_images - 1:
            dx = centroids[i + 1][0] - centroids[i][0]
            dy = centroids[i + 1][1] - centroids[i][1]
            # norm of vector (dx, dy) should be 3
            norm = np.sqrt(dx ** 2 + dy ** 2)
            dx /= norm
            dy /= norm
            dx *= 3
            dy *= 3
            ax[i].arrow(centroids[i][0], centroids[i][1], dx, dy, head_width=1, head_length=1, fc='blue', ec='blue')
        # ax[i].scatter(0, 0, color='blue', marker='o')  # Plot centroids
        ax[i].axis('off')

    t = time.localtime()
    path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
    plt.savefig(path)
    try:
        import wandb
        wandb.log({"generated_images_reference": wandb.Image(path)})
    except Exception as e:
        print(f"Error: {e}")

    if local_computer:
        jjj = 0
        #plt.show()


def plot_heatmap_for_different_PG(S_noise_logarithmic, particle_guidance_factor_logarithmic, safe_num_directions):
    heatmap_data = np.zeros((len(particle_guidance_factor_logarithmic), len(S_noise_logarithmic)))

    # Iterate through the provided values of S_noise and particle_guidance and fill the heatmap matrix
    for j, s_churn in enumerate(S_noise_logarithmic):
        for i, particle_guidance_factor in enumerate(reversed(particle_guidance_factor_logarithmic)):
            # y-axis is reversed to have an increasing values from the bottom to the top
            key = f"S_churn: {s_churn:.2f}, Particle Guidance: {particle_guidance_factor:.2f}"
            if key in safe_num_directions:
                # Calculate the mean value of the directions and store it in the heatmap matrix
                heatmap_data[i, j] = np.mean(safe_num_directions[key])
            else:
                heatmap_data[i, j] = np.nan  # Optional: Assign NaN for missing values

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=10)


    # Set x and y axis labels with corresponding values
    # For particle guidance factor logarithmic (y-axis)
    yticks_labels = [f'{pg:.2f}' if set(str(pg)) <= {'0', '1', '.'} else '' for pg in reversed(particle_guidance_factor_logarithmic)]
    yticks_indices = np.arange(len(particle_guidance_factor_logarithmic))

    # For S_noise_logarithmic (x-axis)
    xticks_labels = [f'{sn:.2f}' if set(str(sn)) <= {'0', '1', '.'} else '' for sn in S_noise_logarithmic]
    xticks_indices = np.arange(len(S_noise_logarithmic))

    # Apply the ticks in the plot
    plt.yticks(yticks_indices, yticks_labels)
    plt.xticks(xticks_indices, xticks_labels)

    plt.colorbar(label='Number of different Digits')
    plt.xlabel('Noise weight of S_churn')
    plt.ylabel('Particle Guidance factor')
    plt.title('Number of different Digits')
    mode ='different digit'
    t = time.localtime()
    path = os.path.join(os.getcwd(), 'out',
                        f"heatmap_important_images_{mode}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
    plt.savefig(path)
    wandb.log({f"final image Histogram {mode}": wandb.Image(path)})
    plt.show()


def plot_images_with_centroids_reference(image, centroids, centroids_reference, indexes=None):
    num_of_images = image.shape[0]
    if num_of_images > 10 and indexes is not None:
        idx_to_plot = indexes[:20] if len(indexes) > 20 else indexes + [i for i in range(10 - len(indexes))]
    elif num_of_images > 10 and indexes is None:
        idx_to_plot = [i for i in range(10)]
    else:
        idx_to_plot = [i for i in range(num_of_images)]

    fig, ax = plt.subplots(1, len(idx_to_plot), figsize=(len(idx_to_plot), 2))

    j = 0
    for i in idx_to_plot:
        ax[j].imshow(image[i, 0, :, :], cmap='gray')
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

        # ax[i].scatter(centroids_reference[0][0], centroids_reference[0][1], color='blue', marker='o')  # Plot reference point
        # ax[i].scatter(centroids_reference[0][0], centroids_reference[1][1], color='green', marker='o')
        ax[j].axis('off')

        # ax[i].scatter(0, 0, color='blue', marker='o')  # Plot centroids
        j += 1
    #save the plot
    # add the out folder to the path
    #  time stamp the file name
    #plt.show()


def make_image_binary(image_tensor):
    # Convert the input PyTorch tensor to a NumPy array and scale it to [0, 255]
    image_numpy = image_tensor.numpy().astype(np.float32)
    scaled_images = (image_numpy * 255).astype(np.uint8)

    # Prepare an array to hold the binary output
    binary_images = np.zeros_like(scaled_images)

    # Apply Otsu's method to each image in the batch
    batch_size, channels, height, width = scaled_images.shape
    for i in range(batch_size):
        for j in range(channels):
            # Apply Otsu's thresholding to each individual 32x32 image
            _, binary_image = cv2.threshold(scaled_images[i, j], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_images[i, j] = binary_image  # Store the result

    # Convert the binary images back to a PyTorch tensor
    binary_images_tensor = torch.tensor(binary_images, dtype=torch.float32)

    return binary_images_tensor


def make_image_background_zero(image_tensor):
    # Convert the input PyTorch tensor to a NumPy array and scale it to [0, 255]
    image_numpy = image_tensor.cpu().numpy().astype(np.float32)
    scaled_images = (image_numpy * 255).astype(np.uint8)

    # Prepare an array to hold the output
    result_images = np.zeros_like(scaled_images)

    # Apply Otsu's method to each image in the batch
    batch_size, channels, height, width = scaled_images.shape
    for i in range(batch_size):
        for j in range(channels):
            # Apply Otsu's thresholding to get the binary mask
            _, binary_mask = cv2.threshold(scaled_images[i, j], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Use the mask to set background to zero and keep foreground values unchanged
            result_images[i, j] = np.where(binary_mask == 0, 0, scaled_images[i, j])

    # Convert the result images back to a PyTorch tensor
    result_images_tensor = torch.tensor(result_images, dtype=torch.float32) / 255.0  # Scale back to [0, 1]

    return result_images_tensor.to(image_tensor.device)


def get_direction_mapping(num_of_directions):
    if num_of_directions == 8:
        direction_mapping = {
            0: "left",
            1: "down-left",
            2: "down",
            3: "down-right",
            4: "right",
            5: "up-right",
            6: "up",
            7: "up-left"
        }
    elif num_of_directions == 4:
        direction_mapping = {
            0: "left",
            1: "down",
            2: "right",
            3: "up"
        }
    elif num_of_directions == 2:
        direction_mapping = {
            0: "down",
            1: "up"
        }
    return direction_mapping

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


def get_true_probability(true_probability, network_pkl, mode, num_of_directions):
    if true_probability is not None:
        p_true = true_probability
    else:
        try:
            p_true = re.search(r'-p(\d+\.\d+)', network_pkl) # Extract the true probability from the filename
            if p_true is not None or p_true == 0:
                p_true = float(p_true.group(1))
            else:
                if mode == 'horizontal':
                    print("!!!No true probability found in the filename. Using 0.5 as the default value!!!")
                    p_true = 0.5
                elif mode == 'circle' and num_of_directions == 4:
                    print("!!!No true probability found in the filename. Using 0.25 as the default value!!!")
                    p_true = 0.25
                elif mode == 'circle' and num_of_directions == 8:
                    print("!!!No true probability found in the filename. Using 0.125 as the default value!!!")
                    p_true = 0.125
                else:
                    p_true = 0.5
        except AttributeError:
            p_true = 0.5
    return p_true
def calculate_centroids(image):
    # Convert the image tensor to NumPy array (if necessary)
    image_np = image.cpu().numpy() if hasattr(image, 'numpy') else image.cpu()

    # Get the number of images
    num_of_images = image_np.shape[0]

    centroids = []

    for i in range(num_of_images):
        img = image_np[i, 0, :, :]
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

def plot_heatmap_of_centroids(centroids):
    # Convert to a NumPy array
    centroids_estimated = np.array(centroids)

    # Create a 2D histogram to represent density
    heatmap, xedges, yedges = np.histogram2d(
        centroids_estimated[:, 0],
        centroids_estimated[:, 1],
        bins=(32, 32),  # You can adjust the number of bins for resolution
        range=[[0, 32], [0, 32]]
    )

    # Plotting the heatmap
    plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest', alpha=0.75)
    plt.colorbar(label='Density')

    # Optional: Overlay the scatter plot to visualize individual points if desired
    plt.scatter(centroids_estimated[:, 0], centroids_estimated[:, 1], color='red', marker='x')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Heatmap of Centroid Density')
    t = time.localtime()
    path = os.path.join(os.getcwd(), 'out',
                        f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
    plt.ylim(0, 32)
    plt.xlim(0, 32)
    plt.savefig(path)
    if not torch.cuda.is_available():
        jjj = 0
        #plt.show()
    plt.close()
    wandb.log({"generated_images_centroids": wandb.Image(path)})

def calculate_cumulative_prob(num_images, prob_estimated, p_true):
    k = num_images * prob_estimated[0]
    cumulative_prob_less = binom.cdf(k, num_images, p_true)
    cumulative_prob_more = 1 - binom.cdf(k - 1, num_images, p_true)
    if prob_estimated[0] > p_true:
        print(f"The cumulative probability of going to right more than {k} times is: {cumulative_prob_more}")
        cum_prob = cumulative_prob_more
    else:
        print(f"The cumulative probability of going to right less than {k} times is: {cumulative_prob_less}")
        cum_prob = cumulative_prob_less
    # log to wandb
    wandb.log({"cumulative_prob": cum_prob})
    wandb.log({"Probability of going to right": prob_estimated[0]})

def plot_images(image):
    num_of_images = image.shape[1]
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))
    for i in range(num_of_images):
        ax[i].imshow(image[0, i, :, :], cmap='gray')
        ax[i].axis('off')
    #plt.show()
    plt.close()

def plot_the_batch_of_generated_images(generated_images, local_computer = False, config_in_title = ""):
    num_of_images = len(generated_images)
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))
    plt.suptitle(f"Generated Images {config_in_title}")
    for i in range(num_of_images):
        ax[i].imshow(generated_images[i,0,:,:], cmap='gray')
        ax[i].axis('off')
    # save the image in out folder with a time stamp
    t = time.localtime()
    path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
    plt.savefig(path)
    # save it to wandb
    import wandb
    wandb.log({"generated_images": wandb.Image(path)})
    if local_computer:
        jjj = 0
        plt.show()
    plt.close()



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
    #plt.show()
    plt.close()

def get_directions(num_of_directions, mode):
    if num_of_directions == 8 and mode == 'circle':
        directions = np.array([
            [-4, 0],  # left
            [-4, 4],  # down-left
            [0, 4],  # down
            [4, 4],  # down-right
            [4, 0],  # right
            [4, -4],  # up-right
            [0, -4],  # up
            [-4, -4]  # up left
        ])
    elif num_of_directions == 4 and mode == 'circle':
        directions = np.array([
            [-4, 0],  # left
            [0, 4],  # down
            [4, 0],  # right
            [0, -4]  # up
        ])
    elif num_of_directions == 2 and mode == 'circle':
        directions = np.array([
            [0, 4],  # Down
            [0, -4]  # Up
        ])
    elif mode == 'horizontal':
        directions = np.array([
            [2, 0],  # Right
            [-2, 0]  # Left
        ])
    return directions

import torch.nn.functional as F
import cv2

def translate_image(image, d_x, d_y):
    M = np.float32([
        [1, 0, d_x],
        [0, 1, d_y]
    ])

    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def calculate_quality_of_images(img_cat_btw_0_1, vectors, conditional_image):
    quality_of_images = []

    # Loop through each image and its corresponding vector
    for i, vector in enumerate(vectors):
        # Ensure vector is a PyTorch tensor
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector, dtype=torch.float32)
        translated = translate_image(conditional_image.squeeze().cpu().numpy(), vector[0].item(), vector[1].item())
        translated = torch.tensor(translated, dtype=torch.float32).unsqueeze(0)
        # Calculate the L2 distance (Euclidean distance) between the moved image and the conditional image
        distance = torch.norm(img_cat_btw_0_1[i, :, :, :] - translated, p=2)
        quality_of_images.append(distance.item())
    return quality_of_images


# ----------------------------- generating function -----------------------------
# ------------------------------------------------------------------------------
def generate_images_and_save_heatmap(
        network_pkl, outdir, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
        ,mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None, s_noise=1
        , gamma_scheduler=False, alpha_scheduler=False,particle_guidance_distance='l2', iteration=0,
        path_classifier=None, separate_grad_and_PG=False, generate_batch_sequentially=False):
    """Generate images with S_churn=0.9 and create a heatmap of pixel intensities."""

    plotting = False
    if local_computer:
        device = torch.device('cpu')

    seeds = [i + 1000 + iteration*num_images for i in range(num_images)]
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    # get the sampler
    sampler = edm_sampler2 if generate_batch_sequentially else edm_sampler

    # ------------------- Initialize the MovingMNIST dataset -------------------
    device_cpu = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    image_sum = None
    j = 0
    generated_images = []
    if local_computer and plotting:
        iterator = tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))
    else:
        iterator = rank_batches
    #  ------------------------- for loop to generate the images -------------------------
    for batch_seeds in iterator:
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        # Generate the image with S_churn=0.9
        generated_img, image_steps = sampler(
            net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, S_churn=S_churn, image=None, plot_diffusion=False, S_noise=s_noise, particle_guidance_factor=particle_guidance_factor,
            gamma_scheduler=gamma_scheduler,particle_guidance_distance=particle_guidance_distance, alpha_scheduler=alpha_scheduler,
            separate_grad_and_PG=separate_grad_and_PG
        )
        generated_img = generated_img.clip(-1, 1)
        generated_img_btw_0_1 = (generated_img + 1) / 2
        generated_img_zero_background = make_image_background_zero(generated_img_btw_0_1)
        batch_predicted_digits, _ = get_prediction(generated_img_zero_background, device_cpu, path_classifier)

        img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        generated_images.append(img_np[0])

        if image_sum is None:
            image_sum = np.sum(img_np.astype(np.float32), axis=0) if img_np.shape[0] > 1 else img_np[0].astype(np.float32)
        else:
            image_sum += np.sum(img_np.astype(np.float32), axis=0) if img_np.shape[0] > 1 else img_np[0].astype(np.float32)

        config_in_title = f"S_churn_{S_churn:.2f}_PG_{particle_guidance_factor:.2f}"
        try:
            plot_the_batch_of_generated_images(generated_images=generated_img_btw_0_1.cpu(), config_in_title = config_in_title,
                                               local_computer=(local_computer))
        except Exception as e:
            print(f"Error: {e}")
        #plot_the_batch_of_generated_images(generated_img_zero_background.cpu(), (local_computer and plotting))


    # Average the pixel intensities to compute the heatmap
    if image_sum.ndim == 2:
        image_sum = np.expand_dims(image_sum, axis=0)
    image_mean = image_sum / num_images
    try:
        generate_heatmap(image_mean, outdir, local_computer=(local_computer and plotting))
    except Exception as e:
        print(f"Error: {e}")
    # batch_predicted_digits to list of integers
    batch_predicted_digits = [int(digit) for digit in batch_predicted_digits]

    if local_computer:
        image_steps_in_big_tensor = image_steps[-1]
        for i in range(1, len(image_steps)):
            image_steps_in_big_tensor = torch.cat((image_steps_in_big_tensor, image_steps[i]), dim=0)
        # trajectory of the diffusion for one generated image
        one_image_diffusion = image_steps_in_big_tensor[0][0].unsqueeze(0).unsqueeze(0)
        for i in range(1, len(image_steps_in_big_tensor)):
            one_image_diffusion = torch.cat((one_image_diffusion, image_steps_in_big_tensor[i][0].unsqueeze(0).unsqueeze(0)), dim=0)

        feature_pwd = os.path.join(os.getcwd(), 'features2.npy')
        labels_pwd = os.path.join(os.getcwd(), 'labels2.npy')
        add_new_unlabeled_images_into_2d_featuremap(images_to_add=generated_img_zero_background,feature_pwd=feature_pwd, labels_pwd=labels_pwd)

        # same with the trajectory of the diffusion
        feature_pwd = os.path.join(os.getcwd(), 'features_noise_anno.npy')
        labels_pwd = os.path.join(os.getcwd(), 'labels_noise_anno.npy')
        #add_new_unlabeled_images_into_2d_featuremap(images_to_add=generated_img_btw_0_1,feature_pwd=feature_pwd, labels_pwd=labels_pwd)



    return batch_predicted_digits

# ----------------------------- main function -----------------------------
# -------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir', help='Where to save the output images and heatmap', metavar='DIR', type=str, required=True)
@click.option('--num_images', help='Number of images to generate', metavar='INT', type=click.IntRange(min=1),
              default=30)
@click.option('--max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=8)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1),
              default=22)
@click.option('--sigma_min', help='Lowest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=0.002)
@click.option('--sigma_max', help='Highest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=80)
@click.option('--s_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5,
              show_default=True)
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=7)
@click.option('--local_computer', help='Use local computer', is_flag=True)
@click.option('--true_probability', help='True probability of going to right', metavar='FLOAT', type=float, default=None)
@click.option('--num_of_directions', help='Number of directions to sample', metavar='INT', type=click.IntRange(min=1),
              default=2)
@click.option('--num_seq', help='Number of sequences', metavar='INT', type=click.IntRange(min=1),
              default=11)
@click.option('--mode', help='Mode of the moving mnist dataset: free, circle, horizontal', metavar='STR', type=str, default='horizontal')
#@click.option('--moving_mnist_path', help='Path to the moving mnist dataset', metavar='STR', type=str, required=True)
@click.option('--particle_guidance_factor', help='Particle guidance factor', metavar='FLOAT', type=click.FloatRange(min=0), default=0)
@click.option('--digit_filter', help='Filter the digit to generate', metavar='INT', type=int, default=None)
@click.option('--pg_heatmap',            help='Generate images and save heatmap',                                 is_flag=True)
@click.option('--gamma_scheduler', help='Use gamma scheduler', is_flag=True)
@click.option('--alpha_scheduler', help='Use alpha scheduler', is_flag=True)
@click.option('--particle_guidance_distance', help='Particle guidance distance, l2 or iou', metavar='STR', type=str, default='l2')
@click.option('--path_classifier', help='Path to the classifier model', metavar='STR', type=str, default='/Users/juliankleutgens/PycharmProjects/edm-main/mnist_resnet18_5e.pth')
@click.option('--separate_grad_and_pg', help='Separate the gradient and particle guidance', is_flag=True)
@click.option('--generate_batch_sequentially', help='Generate the batch sequentially', is_flag=True)

def main(network_pkl, outdir, num_images, max_batch_size, num_steps, sigma_min, sigma_max, s_churn, rho,
         local_computer, true_probability=None, num_seq=1, mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None,
         pg_heatmap = False, gamma_scheduler=False, alpha_scheduler=False, particle_guidance_distance='l2', path_classifier=None,
         separate_grad_and_pg=False, generate_batch_sequentially=False):
    device = torch.device('cpu' if local_computer else 'cuda')
    results = []
    mean_uniform = []
    dist.init()


    wandb.init(project="edm_generation")
    wandb.config.update({"network_pkl": network_pkl, "outdir": outdir, "num_images": num_images, "max_batch_size": max_batch_size,
                         "num_steps": num_steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "S_churn": s_churn, "rho": rho,
                         "local_computer": local_computer, "device": device, "num_seq": num_seq, "mode":mode,
                         "num_of_directions":num_of_directions, "particle_guidance_factor":particle_guidance_factor,
                         "digit_filter":digit_filter, "pg_heatmap":pg_heatmap, "gamma_scheduler":gamma_scheduler, "alpha_scheduler":alpha_scheduler,
                         "particle_guidance_distance":particle_guidance_distance, "path_classifier":path_classifier, "separate_grad_and_PG": separate_grad_and_pg,
                         "generate_batch_sequentially":generate_batch_sequentially})

    digit_prob = {str(i): [] for i in range(10)}
    s_noise = 1

    mode = 'unconditional'
    num_images = 10
    max_batch_size = 10
    S_noise_iterater = [-2, -1.5, -1, -0.5, 0]
    S_noise_logarithmic = 10 ** np.array(S_noise_iterater)
    S_noise_logarithmic = np.insert(S_noise_logarithmic, 0, 0)
    if local_computer:
        S_noise_logarithmic = [0, 1]
    # add zero to the list
    particle_guidance_factor_iterater = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]  # [ 1.5, 2, 2.5, 3]#
    particle_guidance_factor_logarithmic = 10 ** np.array(particle_guidance_factor_iterater)
    #particle_guidance_factor_logarithmic = np.insert(particle_guidance_factor_logarithmic, 0, 0)
    if local_computer:
        particle_guidance_factor_logarithmic = [0, 1]
    num_seq_iter = range(num_seq)

    safe_digits = {}
    count_print = 0 # to count the number of iterations
    for s_churn in S_noise_logarithmic:
        for particle_guidance_factor in particle_guidance_factor_logarithmic:
            print(f"Count: {count_print}")
            for i in num_seq_iter:
                digit = generate_images_and_save_heatmap(
                    network_pkl=network_pkl, outdir=outdir, num_images=num_images, max_batch_size=max_batch_size,
                    num_steps=num_steps, mode=mode, num_of_directions=num_of_directions,
                    sigma_min=sigma_min, sigma_max=sigma_max, S_churn=s_churn, rho=rho, local_computer=local_computer, device=device,
                    particle_guidance_factor=particle_guidance_factor, digit_filter=digit_filter, s_noise=s_noise,
                    gamma_scheduler=gamma_scheduler, alpha_scheduler=alpha_scheduler, particle_guidance_distance=particle_guidance_distance, iteration=i,
                    path_classifier=path_classifier, separate_grad_and_PG=separate_grad_and_pg, generate_batch_sequentially=generate_batch_sequentially,)

                if pg_heatmap:
                    key = f"S_churn: {s_churn:.2f}, Particle Guidance: {particle_guidance_factor:.2f}"
                    if safe_digits.get(key) is None:
                        safe_digits[key] = [len(set(digit))]
                    else:
                        safe_digits[key].append(len(set(digit)))
            count_print += 1



    if pg_heatmap:
        plot_heatmap_for_different_PG(S_noise_logarithmic, particle_guidance_factor_logarithmic, safe_digits)
    results = np.array(results)


if __name__ == "__main__":
    main()
