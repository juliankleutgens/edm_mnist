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
from generate_conditional_frames import edm_sampler, cosine_annealing, plot_diffusion_process_conditional, plot_the_gradient_norm, plot_diffusion_process
import math

# ----------------------------- utility functions -----------------------------
# -----------------------------------------------------------------------------
def generate_heatmap(image_mean, config_in_title,local_computer=False):
    """Generate and save a heatmap based on the mean pixel intensities of generated images."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_mean[:, :, 0], cmap='hot', interpolation='nearest')  # Assuming grayscale images
    plt.colorbar(label='Pixel Intensity')
    plt.title(f'Heatmap of Generated Images with {config_in_title}')

    # save the image in out folder with a time stamp
    t = time.localtime()
    path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
    plt.savefig(path)
    # save it to wandb
    import wandb
    wandb.log({"generated_images_mean": wandb.Image(path)})
    if local_computer:
        jjj = 0
        #plt.show()
    plt.close()

def plot_images_with_centroids(image, centroids, local_computer=False):
    num_of_images = image.shape[0]
    plt.close()
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))

    for i in range(num_of_images):
        ax[i].imshow(image[i, 0, :, :], cmap='gray')
        #ax[i].scatter(centroids[i][0], centroids[i][1], color='red', marker='x')  # Plot centroids
        # add an arrow to show the direction of the centroid
        if i < num_of_images - 1:
            dx = centroids[i + 1][0] - centroids[i][0]
            dy = centroids[i + 1][1] - centroids[i][1]
            # norm of vector (dx, dy) should be 3
            norm = np.sqrt(dx ** 2 + dy ** 2)
            dx /= norm if norm != 0 else 1
            dy /= norm if norm != 0 else 1
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


def plot_heatmap_for_different_PG(S_noise_logarithmic, particle_guidance_factor_logarithmic, safe_num_directions, mode='directions'):
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
    if mode == 'quality' and np.nanmax(heatmap_data) < 20:
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=20)
    elif mode == 'directions' and np.nanmax(heatmap_data) < 8:
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=3, vmax=7)
    else:
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')

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

    # Add labels and title
    if mode == 'directions':
        plt.colorbar(label='Mean Number of Directions')
        plt.xlabel('Noise weight of S_churn')
        plt.ylabel('Particle Guidance factor')
        plt.title('Mean Number of Directions')
    elif mode == 'quality':
        plt.colorbar(label='Mean Quality of Images')
        # scale the colorbar up to 15 when max value is less than 15

        plt.xlabel('Noise weight of S_churn')
        plt.ylabel('Particle Guidance factor')
        plt.title('Mean Quality of Images')

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

def polt_images_highlight_direction_change(image, direction_change, save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
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


"""
        # Clone the image to avoid modifying the original tensor
        moved_image = img_cat_btw_0_1[i, :, :, :].clone()

        # Permute the image to (height, width, channels) format
        moved_image = moved_image.permute(1, 2, 0)
        moved_image = moved_image.float()

        # Create a 2x3 affine matrix for translation
        translation_matrix = torch.eye(2, 3, dtype=torch.float32)  # Identity matrix
        translation_matrix[:, 2] = -vector[:2]  # Apply the translation from the vector (reverse direction)

        # Create the affine grid for transformation
        grid = F.affine_grid(translation_matrix.unsqueeze(0), moved_image.unsqueeze(0).size(),
                             align_corners=False)

        # Apply the translation using grid_sample
        moved_image_shifted = F.grid_sample(
            moved_image.unsqueeze(0),  # Add batch and channel dimensions
            grid,
            align_corners=False
        ).squeeze()  # Remove the added dimensions

        # Calculate the L2 distance (Euclidean distance) between the moved image and the conditional image
        distance = torch.norm(moved_image_shifted - conditional_image, p=2)

        # Append the calculated distance to the quality list
        quality_of_images.append(distance.item())

    return quality_of_images
"""
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


def compute_particle_guidance_grad_set(x, gamma=1, alpha=1, distance='l2', set_reference=None):
    xs = torch.cat((x,set_reference), dim=0) if set_reference is not None else x
    with torch.enable_grad():
        xs = (xs.detach().clone() + 1) * 0.5  # transform from value range [-1,1] to [0,1]
        xs.requires_grad = True
        n = xs.shape[0]

        # Compute matrix of L2 distances
        #  xs.flatten(1).shape = torch.Size([8, 1024])
        distance_matrix = compute_distance_matrix(xs, distance=distance)

        # Only consider upper triangular distance matrix, rest are duplicate entries or distance to self
        triu_indices = torch.triu_indices(n, n, offset=1) # triu_indices = torch.Size([2, 28]) = (n^2 - n) / 2
        distance_list = distance_matrix[triu_indices[0], triu_indices[1]] #  L2 distence

        # Normalizing factor
        h_t = distance_list.median() ** 2 / math.log(n)

        # Sum of RBF kernels
        rbf_sum = (-distance_list * gamma / h_t).exp().sum()
        rbf_sum = rbf_sum * alpha
        rbf_sum.backward()

        xs_grad_torch = xs.grad
        return xs_grad_torch

        """
        xs_grad = torch.zeros_like(xs)
        # do gradient by hand
        counter = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    rbf_weight = (-distance_matrix[i, j] * gamma / h_t).exp()
                    grad_contribution = - 2 * gamma * (xs[i] - xs[j]) * rbf_weight / h_t
                    xs_grad[i] += grad_contribution
                    counter += 1
        xs_grad_torch = xs.grad
        # check if the two ways to calculate the gradient are the same
        if not torch.allclose(xs_grad_torch, xs_grad, atol=1e-2):
            # print how many entries are not the same
            print(f"The two ways to calculate the gradient are not the same. {torch.sum(xs_grad_torch[0, 0, :, :]  - xs_grad[-1, 0, :, :] > 1e-4)}")

        plt.close() 
        plt.imshow(xs_grad_torch[0, 0, :, :], cmap='gray')
        plt.show()
        
        plt.close() 
        plt.imshow(xs[-1, 0, :, :].detach().numpy(), cmap='gray')
        plt.show()
        return xs_grad_torch
        """

def edm_sampler2(
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
    gamma_schedule = [1 for s in range(num_steps)]
    alpha_schedule = [cosine_annealing(s, num_steps, 0, 1) for s in range(num_steps)] if alpha_scheduler else [1 for s in range(num_steps)]

    # plot t steps
    intermediate_images = []
    intermediate_denoised = []
    intermediate_denoised_prime = []
    intermediate_direction_cur = []
    particle_guidance_grad_images = []
    norm_of_gradient_pg = []

    # Main sampling loop.
    gamma_arr = []
    intermediate_images = []
    # make dooble the loop if separate_grad_and_PG is True such that PG and the gradient are added in different steps
    batchsize = latents.shape[0]
    for j in range(batchsize):

        x_next = latents[j].to(torch.float64).unsqueeze(0) * t_steps[0] * 0.1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
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
            # d_cur = (x_hat - denoised) / t_hat
            if not j == 0:
                pg_grad = compute_particle_guidance_grad_set(denoised, gamma=gamma_schedule[i], alpha=alpha_schedule[i],
                                                             distance=particle_guidance_distance, set_reference=set_of_generated_images)
                # norm of the gradient in comparison to the image
                norm_denoised = torch.norm(denoised, p=2)
                pg_grad = pg_grad[0].unsqueeze(0)
                norm_of_gradient_pg.append(torch.norm(pg_grad, p=2))
            else:
                pg_grad = torch.zeros_like(denoised)
            if j == 1:
                denoised_image = (denoised * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                intermediate_denoised.append(denoised_image[0])
            particle_guidance_grad = particle_guidance_factor * t_cur * pg_grad
            if torch.isnan(pg_grad).any() or torch.isinf(pg_grad).any():
                #print('Nan or Inf in pg_grad')
                d_cur = (x_hat - denoised) / t_hat
            else:
                d_cur = (x_hat - denoised) / t_hat - particle_guidance_grad

            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                if net.num_cond_frames > 0:
                    x_input = torch.cat([cond_frames, x_hat], dim=1)
                else:
                    x_input = x_hat
                denoised = net(x_input, t_next, class_labels, num_cond_frames=net.num_cond_frames).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


            if j == 1:
                intermediate_image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3,
                                                                                                 1).cpu().numpy()
                intermediate_images.append(intermediate_image[0])  # Save the first image for plotting

        set_of_generated_images = torch.cat((set_of_generated_images, x_next), dim=0)  if 'set_of_generated_images' in locals() else x_next

        # Save intermediate images.
        # Convert x_next to an image and store it

    if plot_diffusion:
        try:
            plot_the_batch_of_generated_images(generated_images=set_of_generated_images.cpu(),
                                               local_computer=(local_computer))
        except Exception as e:
            print(f"Error: {e}")

        #plot_the_gradient_norm(norm_of_gradient_pg, num_steps)

        #plot_diffusion_process(intermediate_denoised, variable_name='Denoised')
        #plot_diffusion_process(intermediate_direction_cur, variable_name='Direction Cur')
        #plot_diffusion_process_conditional(intermediate_images, images=image)
        #plot_diffusion_process(intermediate_denoised_prime, variable_name='Denoised Prime')
        #plot_diffusion_process(particle_guidance_grad_images, variable_name='Particle Guidance Grad')
    #plot_gamma(gamma_arr, S_churn)
    return set_of_generated_images, intermediate_images




# ----------------------------- generating function -----------------------------
# ------------------------------------------------------------------------------
def generate_images_and_save_heatmap(dataset_obj, dataset_sampler,
        network_pkl, outdir, moving_mnist_path, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
        ,mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None, s_noise=1
        , gamma_scheduler=False, alpha_scheduler=False,particle_guidance_distance='l2', separate_grad_and_PG=False,
        generate_batch_sequentially=False):
    """Generate images with S_churn=0.9 and create a heatmap of pixel intensities."""


    if local_computer:
        device = torch.device('cpu')
    plotting = False

    seeds = [i for i in range(num_images)]
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

    # ------------------- Initialize the MovingMNIST dataset -------------------
    device_cpu = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load the MovingMNIST dataset
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=1))
    image_data, labels, direction_change = next(dataset_iterator)
    image_seq = image_data.to(device)
    image_seq1 = image_seq[:, :, :, :, 0]
    images, labels = convert_video2images_in_batch(images=image_seq, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)

    # ------------------- plot the images with the centroids  -------------------
    if False:
        centroids_seq1 = calculate_centroids(image=image_seq1.permute(1, 0, 2, 3).to(device_cpu))
        plot_images_with_centroids(image=image_seq1.permute(1, 0, 2, 3).to(device), centroids=centroids_seq1,
                                   local_computer=local_computer)

    digit = torch.argmax(labels[0, 0, :]).item() + 1
    images = images.to(device).to(torch.float32) * 2 - 1

    idx = direction_change[:, 1] - net.num_cond_frames + 1
    image = images[int(idx):int(idx) + 1, :, :, :]

    centroids = calculate_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2)
    if (local_computer):
        polt_images_highlight_direction_change(image_data, direction_change)
        try:
            plot_images_with_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2, centroids=centroids,
                                       local_computer=(local_computer and True))
        except Exception as e:
            print(f"Error: {e}")

    #img_cent = (image * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()

    directions = get_directions(num_of_directions, mode)
    # Store sum of images for the heatmap
    sampler = edm_sampler2 if generate_batch_sequentially else edm_sampler
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
        generated_img, _ = sampler(
                net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, S_churn=S_churn, image=image, plot_diffusion=False, S_noise=s_noise,
                particle_guidance_factor=particle_guidance_factor, local_computer=local_computer,
                gamma_scheduler=gamma_scheduler, particle_guidance_distance=particle_guidance_distance,
                alpha_scheduler=alpha_scheduler,
                separate_grad_and_PG=separate_grad_and_PG
            )

        generated_img = generated_img.clip(-1, 1)

        img_btw_0_1 = (generated_img + 1) / 2
        img_cat_btw_0_1 = torch.cat((img_cat_btw_0_1, img_btw_0_1),dim=0) if 'img_cat_btw_0_1' in locals() else img_btw_0_1

        # img_np = (img_btw_0_1 * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        generated_images.append(img_np[0])

        if image_sum is None:
            image_sum = np.sum(img_np.astype(np.float32), axis=0) if img_np.shape[0] > 1 else img_np[0].astype(np.float32)
        else:
            image_sum += np.sum(img_np.astype(np.float32), axis=0) if img_np.shape[0] > 1 else img_np[0].astype(np.float32)


    # Average the pixel intensities to compute the heatmap
    if image_sum.ndim == 2:
        image_sum = np.expand_dims(image_sum, axis=0)
    image_mean = image_sum / num_images
    try:
        config_in_title = f"S_churn_{S_churn:.2f}_PG_{particle_guidance_factor:.2f}"
        generate_heatmap(image_mean=image_mean,config_in_title=config_in_title , local_computer=(local_computer))
    except Exception as e:
        print(f"Error: {e}")


    centroids_estimated = calculate_centroids(image=img_cat_btw_0_1)
    estimated_directions = []
    vectors = []
    i = 0
    for c in centroids_estimated:
        x_t_prev = centroids[-2][0]
        y_t_prev = centroids[-2][1]
        vector = np.array([c[0] - x_t_prev, c[1] - y_t_prev])
        vectors.append(vector)
        vector_norm = vector / np.linalg.norm(vector)
        directions_norm = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        cosine_similarities = np.dot(directions_norm, vector_norm)
        closest_index = np.argmax(cosine_similarities)
        estimated_directions.append(closest_index)

    last_conditional_image_btw_0_1 = (image[:, -2, :, :] + 1) / 2
    qualtity_of_images = calculate_quality_of_images(img_cat_btw_0_1.cpu(), vectors, last_conditional_image_btw_0_1.cpu())
    mean_quality = np.mean(qualtity_of_images)


    count = dict(Counter(estimated_directions))
    prob_estimated = {k: v / num_images for k, v in count.items()}

    if local_computer and plotting:
        plot_images_with_centroids_reference(image=img_cat_btw_0_1, centroids=centroids_estimated, centroids_reference=centroids[-2:])
    return prob_estimated, digit, mean_quality

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
@click.option('--moving_mnist_path', help='Path to the moving mnist dataset', metavar='STR', type=str, required=True)
@click.option('--particle_guidance_factor', help='Particle guidance factor', metavar='FLOAT', type=click.FloatRange(min=0), default=0)
@click.option('--digit_filter', help='Filter the digit to generate', metavar='INT', type=int, default=None)
@click.option('--pg_heatmap',            help='Generate images and save heatmap',                                 is_flag=True)
@click.option('--gamma_scheduler', help='Use gamma scheduler', is_flag=True)
@click.option('--alpha_scheduler', help='Use alpha scheduler', is_flag=True)
@click.option('--particle_guidance_distance', help='Particle guidance distance, l2 or iou', metavar='STR', type=str, default='l2')
@click.option('--separate_grad_and_pg', help='Separate gradient and particle guidance', is_flag=True)
@click.option('--generate_batch_sequentially', help='Generate the batch sequentially', is_flag=True)

def main(network_pkl, outdir, num_images, max_batch_size, num_steps, sigma_min, sigma_max, s_churn, rho,moving_mnist_path,
         local_computer, true_probability=None, num_seq=1, mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None,
         pg_heatmap = False, gamma_scheduler=False, alpha_scheduler=False, particle_guidance_distance='l2', separate_grad_and_pg=False,
         generate_batch_sequentially=False):
    device = torch.device('cpu' if local_computer else 'cuda')
    results = []
    mean_uniform = []
    dist.init()

    p_true = get_true_probability(true_probability, network_pkl, mode, num_of_directions)
    if mode == 'circle' and num_of_directions == 4 or num_of_directions == 8:
        direction_mapping = get_direction_mapping(num_of_directions)
        stored_probabilities = {key: [] for key in direction_mapping.values()}


    wandb.init(project="edm_generation")
    wandb.config.update({"network_pkl": network_pkl, "outdir": outdir, "num_images": num_images, "max_batch_size": max_batch_size,
                         "num_steps": num_steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "S_churn": s_churn, "rho": rho,
                         "local_computer": local_computer, "device": device, "true_probability": p_true, "num_seq": num_seq, "mode":mode,
                         "num_of_directions":num_of_directions, "particle_guidance_factor":particle_guidance_factor,
                         "digit_filter":digit_filter, "pg_heatmap":pg_heatmap, "gamma_scheduler":gamma_scheduler, "alpha_scheduler":alpha_scheduler,
                         "particle_guidance_distance":particle_guidance_distance})

    digit_prob = {str(i): [] for i in range(10)}
    if digit_filter is not None:
        digit_filter = [digit_filter]
    s_noise = 1
    if pg_heatmap:
        mode = 'circle'
        num_images = num_of_directions
        S_noise_iterater = [-2, -1.5, -1, -0.5, 0]
        S_noise_logarithmic = 10 ** np.array(S_noise_iterater)
        S_noise_logarithmic = np.insert(S_noise_logarithmic, 0, 0)
        if local_computer:
            S_noise_logarithmic = [0]
        # add zero to the list
        particle_guidance_factor_iterater = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75 , 1]#[ 1.5, 2, 2.5, 3]#
        particle_guidance_factor_logarithmic = 10 ** np.array(particle_guidance_factor_iterater)
        #particle_guidance_factor_logarithmic = [10, 100, 1000, 10000]
        #particle_guidance_factor_logarithmic = np.insert(particle_guidance_factor_logarithmic, 0, 0)
        if local_computer:
            particle_guidance_factor_logarithmic = [1]
        num_seq_iter = range(num_seq)

    else:
        S_noise_logarithmic = [s_churn]
        particle_guidance_factor_logarithmic = [particle_guidance_factor]
        num_seq_iter = tqdm(range(num_seq), desc="Generating images")
    safe_num_directions = {}
    safe_quality = {}
    count_print = 0 # to count the number of iterations

    generation_kwargs = {
        "network_pkl": network_pkl,
        "outdir": outdir,
        "num_images": num_images,
        "max_batch_size": max_batch_size,
        "num_steps": num_steps,
        "mode": mode,
        "num_of_directions": num_of_directions,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
        "local_computer": local_computer,
        "device": device,
        "digit_filter": digit_filter,
        "s_noise": s_noise,
        "gamma_scheduler": gamma_scheduler,
        "alpha_scheduler": alpha_scheduler,
        "particle_guidance_distance": particle_guidance_distance,
        "separate_grad_and_PG": separate_grad_and_pg,
        "generate_batch_sequentially": generate_batch_sequentially
    }
    mean_quality_list = []
    for s_churn in S_noise_logarithmic:
        for particle_guidance_factor in particle_guidance_factor_logarithmic:
            # we want to have the same number of images for each case
            count_print += 1
            dataset_obj = MovingMNIST(train=True, data_root=moving_mnist_path, seq_len=32, num_digits=1, image_size=32,
                                      mode=mode,
                                      deterministic=False, log_direction_change=True, step_length=0.1,
                                      let_last_frame_after_change=False, use_label=True,
                                      num_of_directions_in_circle=num_of_directions, digit_filter=digit_filter)
            dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
            print(f"Count: {count_print}")
            for i in num_seq_iter:
                generation_kwargs.update({
                    "S_churn": s_churn,
                    "particle_guidance_factor": particle_guidance_factor,
                })

                prob_estimated, digit, mean_quality = generate_images_and_save_heatmap(dataset_obj=dataset_obj, dataset_sampler=dataset_sampler,
                    moving_mnist_path=moving_mnist_path, **generation_kwargs)

                if pg_heatmap:
                    key = f"S_churn: {s_churn:.2f}, Particle Guidance: {particle_guidance_factor:.2f}"
                    if safe_num_directions.get(key) is None:
                        safe_num_directions[key] = [len(prob_estimated)]
                    else:
                        safe_num_directions[key].append(len(prob_estimated))
                    if safe_quality.get(key) is None:
                        safe_quality[key] = [mean_quality]
                    else:
                        safe_quality[key].append(mean_quality)

                else:
                    if not 0 in prob_estimated and not pg_heatmap:
                        prob_estimated[0] = 0
                    results.append(prob_estimated[0])
                    mean_quality_list.append(mean_quality)
                    wandb.log({"Mean Quality": np.mean(np.array(mean_quality))})
                    wandb.log({"Mean": np.mean(np.array(results))})

                if mode == 'horizontal' and not pg_heatmap:
                    calculate_cumulative_prob(num_images, prob_estimated, p_true)

                if mode == 'circle' and (num_of_directions == 4 or num_of_directions == 8) and not pg_heatmap:
                    mean_uniform.append(np.mean([value for key, value in prob_estimated.items() if key != 0]))
                    wandb.log({f"Mean of the other directions which are uniformed, should: {((1-p_true)/(num_of_directions-1)):.2f}": np.mean(mean_uniform)})

                    prob_estimated_with_directions = {direction_mapping[k]: v for k, v in prob_estimated.items()}
                    for key, value in prob_estimated_with_directions.items():
                        stored_probabilities[key].append(value)
                        wandb.log({f"Mean of going to {key}": np.mean(stored_probabilities[key])})
                    wandb.log(prob_estimated_with_directions)

                try:
                    digit_prob[str(digit)].append(prob_estimated[0])
                except KeyError:
                    continue

    if pg_heatmap:
        plot_heatmap_for_different_PG(S_noise_logarithmic, particle_guidance_factor_logarithmic, safe_num_directions)
        plot_heatmap_for_different_PG(S_noise_logarithmic, particle_guidance_factor_logarithmic, safe_quality, mode='quality')
    results = np.array(results)
    mean_value = np.mean(results)
    median_value = np.median(results)
    variance = np.var(results)

    # calculate how possible the sampled probability is to be the same as the real probability
    import scipy.stats as stats

    # calculate the mean, median, and variance of the results
    print(f"Prob for each digit: {digit_prob}")
    print(f"Results: {results}")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Variance: {variance}")

    try:
        # make a plot of the results for each digit
        plt.figure(figsize=(10, 6))
        for key, values in digit_prob.items():
            plt.plot(values, label=f'Digit {key}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Random Values for Each Digit')
        plt.legend()
        # save the plt
        t = time.localtime()
        path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
        plt.savefig(path)
        wandb.log({"final image overview": wandb.Image(path)})
    except Exception as e:
        print(f"Error: {e}")

    try:
        #make a histogram of the for each probability in the digit_prob
        rows = num_of_directions // 2
        fig, axs = plt.subplots(rows, 2, figsize=(10, 10)) # 4 rows, 2 columns
        axs = axs.flatten()
        i = 0
        for keys, values_iteration in stored_probabilities.items():
            if len(values_iteration) < num_images:
                values_iteration = values_iteration.extend([0] * (num_images - len(values_iteration)))
            axs[i].hist(values_iteration, bins=50, alpha=0.5)
            axs[i].set_title(f"Probability of going to {keys}")
            i += 1
        plt.tight_layout()
        #plt.show()
        # save the plt
        t = time.localtime()
        path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}_{random.randint(0, 1000)}.png")
        plt.savefig(path)
        wandb.log({"final image Histogram": wandb.Image(path)})
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
