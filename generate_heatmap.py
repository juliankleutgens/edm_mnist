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



# ----------------------------- utility functions -----------------------------
# -----------------------------------------------------------------------------
def plot_images_with_centroids(image, centroids):
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
    plt.show()


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


def calculate_centroids(image):
    # Convert the image tensor to NumPy array (if necessary)
    image_np = image.numpy() if hasattr(image, 'numpy') else image

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






# ----------------------------- generating function -----------------------------
# ------------------------------------------------------------------------------
def generate_images_and_save_heatmap(
        network_pkl, outdir, num_images=100, max_batch_size=1, num_steps=18,
        sigma_min=0.002, sigma_max=80, S_churn=0.9, rho=7, local_computer=False, device=torch.device('cuda')
):
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


    # Initialize the MovingMNIST dataset
    device_cpu = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)


    dataset_obj = MovingMNIST(train=True, data_root='./data', seq_len=32, num_digits=1, image_size=32, move_horizontally=True,
                              deterministic=False, log_direction_change=True, step_length=0.1, let_last_frame_after_change=False)
    dataset_sampler = torch.utils.data.SequentialSampler(dataset_obj)
    dataset_sampler = RandomSampler(dataset_obj)
    dataset_iterator = iter(DataLoader(dataset_obj, sampler=dataset_sampler, batch_size=1))
    image_data, labels, direction_change = next(dataset_iterator)
    image_seq = image_data.to(device)
    image_seq1 = image_seq[:,:,:,:,0]
    images, labels = convert_video2images_in_batch(images=image_seq, labels=labels, use_label=False,
                                                   num_cond_frames=net.num_cond_frames)

    #centroids = calculate_centroids(image=image_seq1.permute(1, 0, 2, 3).to(device_cpu) )
    #plot_images_with_centroids(image=image_seq1.permute(1, 0, 2, 3).to(device) , centroids=centroids)
    images = images.to(device).to(torch.float32) * 2 - 1
    idx = direction_change[:,1] - net.num_cond_frames + 1
    image = images[int(idx):int(idx)+1, :, :, :]

    centroids = calculate_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2)
    if local_computer:
        polt_images_highlight_direction_change(image_data, direction_change)
        plot_images_with_centroids(image=(image.permute(1, 0, 2, 3).to(device_cpu) + 1) / 2, centroids=centroids)

    #img_cent = (image * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()



    # Store sum of images for the heatmap
    image_sum = None
    j = 0
    generated_images = []
    for batch_seeds in (tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
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
    generate_heatmap(image_mean, outdir, local_computer=local_computer)


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
    result = num_times_went_to_right/num_images

    if local_computer:
        plot_images_with_centroids_reference(image=img_cat_btw_0_1, centroids=centroids_all, centroids_reference=centroids[-2:], indexes=went_to_left)
    return result
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




# ----------------------------- main function -----------------------------
# -------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir', help='Where to save the output images and heatmap', metavar='DIR', type=str, required=True)
@click.option('--num_images', help='Number of images to generate', metavar='INT', type=click.IntRange(min=1),
              default=2)
@click.option('--max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=8)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1),
              default=22)
@click.option('--sigma_min', help='Lowest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=0.002)
@click.option('--sigma_max', help='Highest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=20)
@click.option('--s_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5,
              show_default=True)
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=7)
@click.option('--local_computer', help='Use local computer', is_flag=True)
@click.option('--true_probability', help='True probability of going to right', metavar='FLOAT', type=float, default=None)
@click.option('--num_seq', help='Number of sequences', metavar='INT', type=click.IntRange(min=1), default=2)

def main(network_pkl, outdir, num_images, max_batch_size, num_steps, sigma_min, sigma_max, s_churn, rho,
         local_computer, true_probability=None, num_seq=1):
    device = torch.device('cpu' if local_computer else 'cuda')
    results = []
    dist.init()

    if true_probability is not None:
        p_true = true_probability
    else:
        try:
            p_true = re.search(r'-p(\d+\.\d+)', network_pkl) # Extract the true probability from the filename
            if p_true is not None:
                p_true = float(p_true.group(1))
            else:
                print("!!!No true probability found in the filename. Using 0.5 as the default value!!!")
                p_true = 0.5
        except AttributeError:
            p_true = 0.5

    import wandb
    wandb.init(project="edm_generation")
    wandb.config.update({"network_pkl": network_pkl, "outdir": outdir, "num_images": num_images, "max_batch_size": max_batch_size,
                         "num_steps": num_steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "S_churn": s_churn, "rho": rho,
                         "local_computer": local_computer, "device": device})

    for i in tqdm(range(num_seq), desc="Generating images"):
        result = generate_images_and_save_heatmap(
            network_pkl=network_pkl, outdir=outdir, num_images=num_images, max_batch_size=max_batch_size,
            num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, S_churn=s_churn, rho=rho, local_computer=local_computer, device=device
        )
        k = num_images * result
        cumulative_prob_less = binom.cdf(k, num_images, p_true)
        cumulative_prob_more = 1 - binom.cdf(k-1, num_images, p_true)
        if result > p_true:
            print(f"The cumulative probability of going to right more than {k} times is: {cumulative_prob_more}")
            cum_prob = cumulative_prob_more
        else:
            print(f"The cumulative probability of going to right less than {k} times is: {cumulative_prob_less}")
            cum_prob = cumulative_prob_less
        # log to wandb
        wandb.log({"cumulative_prob_less": cum_prob})
        wandb.log({"result": result})
        wandb.log({"Mean": np.mean(np.array(results))})
        results.append(result)





    results = np.array(results)
    mean_value = np.mean(results)
    median_value = np.median(results)
    variance = np.var(results)

    # calculate how possible the sampled probability is to be the same as the real probability
    import scipy.stats as stats

    # calculate the mean, median, and variance of the results
    print(f"Results: {results}")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Variance: {variance}")


if __name__ == "__main__":
    main()
