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
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from scipy.stats import binom
import time
from collections import Counter



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
    path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.png")
    plt.savefig(path)
    # save it to wandb
    import wandb
    wandb.log({"generated_images": wandb.Image(path)})
    if local_computer:
        plt.show()
    plt.close()

def plot_images_with_centroids(image, centroids, local_computer=False):
    num_of_images = image.shape[0]
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
    path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.png")
    plt.savefig(path)

    import wandb
    wandb.log({"generated_images": wandb.Image(path)})

    if local_computer:
        plt.show()
    plt.close()


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
    plt.show()

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


def plot_images(image):
    num_of_images = image.shape[1]
    fig, ax = plt.subplots(1, num_of_images, figsize=(num_of_images, 2))
    for i in range(num_of_images):
        ax[i].imshow(image[0, i, :, :], cmap='gray')
        ax[i].axis('off')
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
    plt.show()
    plt.close()






# ----------------------------- generating function -----------------------------
# ------------------------------------------------------------------------------
def generate_images_and_save_heatmap(
        network_pkl, outdir, moving_mnist_path, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
        ,mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None):
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
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    dataset_obj = MovingMNIST(train=True, data_root=moving_mnist_path, seq_len=32, num_digits=1, image_size=32, mode=mode,
                              deterministic=False, log_direction_change=True, step_length=0.1, let_last_frame_after_change=False, use_label=True,
                              num_of_directions_in_circle=num_of_directions,digit_filter=None)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    dataset_sampler = RandomSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=1))
    image_data, labels, direction_change = next(dataset_iterator)
    image_seq = image_data.to(device)
    image_seq1 = image_seq[:,:,:,:,0]
    images, labels = convert_video2images_in_batch(images=image_seq, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)



    # ------------------- plot the images with the centroids  -------------------
    if False:
        centroids = calculate_centroids(image=image_seq1.permute(1, 0, 2, 3).to(device_cpu))
        plot_images_with_centroids(image=image_seq1.permute(1, 0, 2, 3).to(device) , centroids=centroids, local_computer=local_computer)
    digit = torch.argmax(labels[0, 0, :]).item() + 1
    images = images.to(device).to(torch.float32) * 2 - 1

    idx = direction_change[:,1] - net.num_cond_frames + 1
    image = images[int(idx):int(idx)+1, :, :, :]

    centroids = calculate_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2)
    if local_computer:
        polt_images_highlight_direction_change(image_data, direction_change)
    plot_images_with_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2, centroids=centroids, local_computer=local_computer)

    #img_cent = (image * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()

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

    # Store sum of images for the heatmap
    #
    image_sum = None
    j = 0
    generated_images = []
    if local_computer:
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
        generated_img, _ = edm_sampler(
            net=net, latents=latents, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, S_churn=S_churn, image=image, plot_diffusion=False
        )


        img_btw_0_1 = (generated_img + 1) / 2
        # thearshold all values below 0.1 to 0
        # img_btw_0_1[img_btw_0_1 < 0.55] = 0
        img_cat_btw_0_1 = torch.cat((img_cat_btw_0_1, img_btw_0_1),dim=0) if 'img_cat_btw_0_1' in locals() else img_btw_0_1

        # img_np = (img_btw_0_1 * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        img_np = (generated_img * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        generated_images.append(img_np[0])

        if image_sum is None:
            image_sum = np.sum(img_np.astype(np.float32), axis=0) if img_np.shape[0] > 1 else img_np[0].astype(np.float32)
        else:
            image_sum += np.sum(img_np.astype(np.float32), axis=0) if img_np.shape[0] > 1 else img_np[0].astype(np.float32)


    # Average the pixel intensities to compute the heatmap
    if image_sum.ndim  == 2:
        image_sum = np.expand_dims(image_sum, axis=0)
    image_mean = image_sum / num_images
    try:
        generate_heatmap(image_mean, outdir, local_computer=local_computer)
    except Exception as e:
        print(f"Error: {e}")


    centroids_all = calculate_centroids(image=img_cat_btw_0_1)
    estimated_directions = []
    i = 0
    for c in centroids_all:
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
    try:
        if mode == 'horizontal':
            print(f"The probability of going to right is: {prob_estimated[0]}")
            print(f"The digit moved to number of times to right is: {count[0]} out of {num_images}")
        elif mode == 'circle' and num_of_directions == 4 or num_of_directions == 8:
            print(f"The probability of going to left is: {prob_estimated[0]}")
            print(f"The digit moved to number of times to left (stronger probability) is: {count[0]} out of {num_images}")
            mean = np.mean([value for key, value in prob_estimated.items() if key != 0])
            print(f"The mean of the other directions is: {mean}")
    except KeyError:
        print(f"KeyError: {count}")



    if local_computer:
        plot_images_with_centroids_reference(image=img_cat_btw_0_1, centroids=centroids_all, centroids_reference=centroids[-2:])
    return prob_estimated, digit

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
              default=8)
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

def main(network_pkl, outdir, num_images, max_batch_size, num_steps, sigma_min, sigma_max, s_churn, rho,moving_mnist_path,
         local_computer, true_probability=None, num_seq=1, mode='horizontal', num_of_directions=2, particle_guidance_factor=0, digit_filter=None):
    device = torch.device('cpu' if local_computer else 'cuda')
    results = []
    mean_uniform = []
    dist.init()

    p_true = get_true_probability(true_probability, network_pkl, mode, num_of_directions)

    import wandb
    wandb.init(project="edm_generation")
    wandb.config.update({"network_pkl": network_pkl, "outdir": outdir, "num_images": num_images, "max_batch_size": max_batch_size,
                         "num_steps": num_steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "S_churn": s_churn, "rho": rho,
                         "local_computer": local_computer, "device": device, "true_probability": p_true, "num_seq": num_seq, "mode":mode,
                         "num_of_directions":num_of_directions, "particle_guidance_factor":particle_guidance_factor})

    digit_prob = {str(i): [] for i in range(10)}
    for i in tqdm(range(num_seq), desc="Generating images"):
        prob_estimated, digit = generate_images_and_save_heatmap(
            network_pkl=network_pkl, outdir=outdir, num_images=num_images, max_batch_size=max_batch_size,
            num_steps=num_steps, mode=mode, num_of_directions=num_of_directions,
            sigma_min=sigma_min, sigma_max=sigma_max, S_churn=s_churn, rho=rho, local_computer=local_computer, device=device, moving_mnist_path=moving_mnist_path,
            particle_guidance_factor=particle_guidance_factor, digit_filter=digit_filter
        )
        if not 0 in prob_estimated:
            prob_estimated[0] = 0
        results.append(prob_estimated[0])

        if mode == 'horizontal':
            k = num_images * prob_estimated[0]
            cumulative_prob_less = binom.cdf(k, num_images, p_true)
            cumulative_prob_more = 1 - binom.cdf(k-1, num_images, p_true)
            if  prob_estimated[0] > p_true:
                print(f"The cumulative probability of going to right more than {k} times is: {cumulative_prob_more}")
                cum_prob = cumulative_prob_more
            else:
                print(f"The cumulative probability of going to right less than {k} times is: {cumulative_prob_less}")
                cum_prob = cumulative_prob_less
            # log to wandb
            wandb.log({"cumulative_prob": cum_prob})
        wandb.log({"Mean": np.mean(np.array(results))})

        if mode == 'circle' and num_of_directions == 4 or num_of_directions == 8:
            direction_mapping = get_direction_mapping(num_of_directions)
            mean_uniform.append(np.mean([value for key, value in prob_estimated.items() if key != 0]))
            wandb.log({f"Mean of the other directions which are uniformed, should: {1-p_true}": np.mean(mean_uniform)})

            prob_estimated_with_directions = {direction_mapping[k]: v for k, v in prob_estimated.items()}
            wandb.log(prob_estimated_with_directions)


        results.append(prob_estimated[0])
        try:
            digit_prob[str(digit)].append(prob_estimated[0])
        except KeyError:
            print(f"KeyError: {digit}")
            continue


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
        plt.figure(figsize=(10, 6))

        for key, values in digit_prob.items():
            plt.plot(values, label=f'Digit {key}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Random Values for Each Digit')
        plt.legend()
        # save the plt
        t = time.localtime()
        path = os.path.join(os.getcwd(), 'out', f"generated_images_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.png")
        plt.savefig(path)
        wandb.log({"final image overview": wandb.Image(path)})
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
