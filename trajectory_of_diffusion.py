import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from moving_mnist import MovingMNIST
from resnet_classifier import get_prediction, add_noise_to_image_like_diffusion
import os

# Plotting the 2D visualization
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


def plot_2d(features_2d, labels, title, new_points=None):
    num_labels = len(np.unique(labels))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

    if new_points is not None:
        for i in range(new_points.shape[0]):
            plt.scatter(new_points[i, 0], new_points[i, 1], c='red', label=f'New Point {i + 1}', marker='x')
    plt.colorbar(scatter, ticks=range(num_labels))  # Assuming labels are integers in range(0, 9)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def get_the_features_and_the_2d_map(num_classes=10):
    # Load your dataset
    moving_mnist_path = '/Users/juliankleutgens/data_edm/data/MNIST'
    dataset_obj = MovingMNIST(train=True, data_root=moving_mnist_path, seq_len=1, num_digits=1, image_size=32,
                              mode='free',
                              deterministic=False, log_direction_change=True, step_length=0.1,
                              let_last_frame_after_change=False, use_label=True)

    dataset_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=6000, shuffle=True)
    images, labels, _ = next(iter(dataset_loader))
    images = images.squeeze(-1)  # Remove the channel dimension

    batch_seeds = [i for i in range(300)]
    rnd = StackedRandomGenerator('cpu', batch_seeds)

    clear_images = False
    if num_classes == 11:
        labels = torch.cat((labels, torch.zeros(labels.size(0), 1, 1)), dim=2) if num_classes == 11 else labels
        if clear_images:
            latents = rnd.randn([len(batch_seeds), 1, 32, 32], device='cpu')
            images = torch.cat((images, latents), dim=0)

            labels_for_latents = torch.zeros(latents.size(0), 1, 11)
            labels_for_latents[:, :, 10] = labels_for_latents[:, :, 10] + 1
            labels = torch.cat((labels, labels_for_latents), dim=0)
        else:
            images, labels = add_noise_to_image_like_diffusion(images, labels)

    # Extract features
    try:
        features_np = np.load('/Users/juliankleutgens/PycharmProjects/edm-main/features2.npy')
        labels_np = np.load('/Users/juliankleutgens/PycharmProjects/edm-main/labels2.npy')
        print("Features and labels loaded from disk")
        get_new_features = False
    except FileNotFoundError:
        get_new_features = True
        print("Extracting features")
    if get_new_features:
        with torch.no_grad():
        # If images are grayscale, convert to 3 channels
            _, features = get_prediction(images, device='cpu',num_classes=num_classes,
                                        path='/Users/juliankleutgens/PycharmProjects/edm-main/mnist_resnet18_noise_class.pth')
        # Convert features and labels to numpy
        features_np = features.numpy()
        labels = torch.argmax(labels, dim=-1)
        labels_np = labels.numpy()

        # save labels and features
        np.save('/Users/juliankleutgens/PycharmProjects/edm-main/features_noise_anno.npy', features_np)
        np.save('/Users/juliankleutgens/PycharmProjects/edm-main/labels_noise_anno.npy', labels_np)




    # Use UMAP or t-SNE for dimensionality reduction
    # Choose either UMAP or t-SNE

    # Option 1: t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    features_2d_tsne = tsne.fit_transform(features_np)

    # Option 2: UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d_umap = umap_reducer.fit_transform(features_np)


    # Now call this function with the correct shapes
    plot_2d(features_2d_tsne, labels_np, title="t-SNE Visualization of MovingMNIST")
    plot_2d(features_2d_umap, labels_np, title="UMAP Visualization of MovingMNIST")

def add_new_unlabeled_images_into_2d_featuremap(images_to_add, labels_to_add=None, feature_pwd=os.path.join(os.getcwd(), 'features2.npy'),
                                         labels_pwd=os.path.join(os.getcwd(), 'labels2.npy')):
    # Load the features and labels
    features = np.load(feature_pwd)
    labels = np.load(labels_pwd)

    # Extract features for the new images
    with torch.no_grad():
        _, new_features = get_prediction(images_to_add, device='cpu',num_classes=10,
        path='/Users/juliankleutgens/PycharmProjects/edm-main/mnist_resnet18_5e.pth')

    # Convert new features to numpy
    new_features_np = new_features.numpy()

    # Concatenate the new features to the existing ones
    features_combined = np.concatenate((features, new_features_np), axis=0)

    # Use the same UMAP or t-SNE model to transform the new points
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d_combined = umap_reducer.fit_transform(features_combined)

    # labels
    # add a new label for the new points
    if labels_to_add is None:
        labels = np.concatenate((labels, np.full(new_features.shape[0], labels.max() + 1).reshape(-1, 1)))


    # Plot the original features and the new points
    new_points_2d = features_2d_combined[-new_features_np.shape[0]:]
    plot_2d(features_2d_combined, labels, title="UMAP Visualization with New Points", new_points=new_points_2d)



if __name__ == '__main__':
    get_the_features_and_the_2d_map(num_classes=11)
    batch_seeds = [i for i in range(100)]
    rnd = StackedRandomGenerator('cpu', batch_seeds)
    latents = rnd.randn([len(batch_seeds), 1, 32, 32], device='cpu')
    # Add new images and labels to the existing feature map
    images_to_add = latents
    features, labels = add_new_unlabeled_images_into_2d_featuremap(images_to_add=images_to_add)
    # print(features.shape, labels.shape)