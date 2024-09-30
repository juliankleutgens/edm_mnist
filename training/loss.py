# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import wandb
import os

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".
def plot_batch_of_image_and_noise(x_batch, n_batch, y_plus_n_batch, Dy_batch, sigma, path=None):
    import matplotlib.pyplot as plt
    import torch

    num_images = 10
    if x_batch.shape[0] < 10:
        num_images = x_batch.shape[0]
    fig, ax = plt.subplots(4, num_images, figsize=(20, 6))

    for i in range(num_images):
        # Handle x_batch (detach if requires grad, and convert to numpy)
        if isinstance(x_batch[i], torch.Tensor):
            x = x_batch[i].detach().clone().to('cpu').numpy()  # Detach to avoid the gradient error
        else:
            x = x_batch[i]  # If it's already a numpy array, use it directly
        x = (x * 127.5 + 128).clip(0, 255).astype('uint8')

        if isinstance(y_plus_n_batch[i], torch.Tensor):
            y_plus_n = y_plus_n_batch[i].detach().clone().to('cpu').numpy()
        else:
            y_plus_n = y_plus_n_batch[i]
        y_plus_n = (y_plus_n * 127.5 + 128).clip(0, 255).astype('uint8')

        # Handle n_batch (detach if requires grad, and convert to numpy)
        if isinstance(n_batch[i], torch.Tensor):
            n = n_batch[i].detach().clone().to('cpu').numpy()  # Detach to avoid the gradient error
        else:
            n = n_batch[i]  # If it's already a numpy array, use it directly
        n = (n * 127.5 + 128).clip(0, 255).astype('uint8')

        # Handle Dy_batch (detach if requires grad, and convert to numpy)
        if isinstance(Dy_batch[i], torch.Tensor):
            Dy = Dy_batch[i].detach().clone().to('cpu').numpy()  # Detach to avoid the gradient error
        else:
            Dy = Dy_batch[i]  # If it's already a numpy array, use it directly
        Dy = (Dy * 127.5 + 128).clip(0, 255).astype('uint8')

        # Plotting
        ax[0, i].imshow(x.squeeze(), cmap='gray')
        ax[0, i].set_title('Image')
        ax[0, i].axis('off')

        ax[1, i].imshow(n.squeeze(), cmap='gray')
        ax[1, i].set_title(f'Noise with sigma={sigma[i].squeeze():.2f}')
        ax[1, i].axis('off')

        ax[2, i].imshow(y_plus_n.squeeze(), cmap='gray')
        ax[2, i].set_title('Image + Noise')
        ax[2, i].axis('off')

        ax[3, i].imshow(Dy.squeeze(), cmap='gray')
        ax[3, i].set_title('D(y+n)')
        ax[3, i].axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        full_path = path + '.png'
        wandb_image = wandb.Image(full_path, caption=f"Image, Noise, Image+Noise, D(y+n)")
        wandb.log({"Image, Noise, Image+Noise, D(y+n)": wandb_image})
    else:
        plt.show()
    plt.close()
    print(f'Saved Image, Noise. W&B run URL is: {wandb.run.get_url()}')

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, plot_batch= True, path=None, num_cond_frames=0):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        if num_cond_frames > 0:
            y_condtional = y[:, :num_cond_frames, :, :]
            y = y[:, num_cond_frames:, :, :]
            n = n[:, num_cond_frames:, :, :]
            y_plus_n = torch.cat([y_condtional, y + n], dim=1)
        else:
            y_plus_n = y + n
        D_yn = net(y_plus_n, sigma, labels, augment_labels=augment_labels, num_cond_frames=num_cond_frames)
        if plot_batch:
            plot_batch_of_image_and_noise(y,n, y+n, D_yn, sigma, path=path)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
