import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from moving_mnist import MovingMNIST  # Assuming you have a MovingMNIST loader
from tqdm import trange


def plot_batch_of_image_and_noise(x_batch, n_batch, y_plus_n_batch, sigma):
    import matplotlib.pyplot as plt
    import torch

    num_images = 8
    if x_batch.shape[0] < num_images:
        num_images = x_batch.shape[0]
    fig, ax = plt.subplots(3, num_images, figsize=(20, 6))

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
    plt.show()





def transform_and_resize_on_cpu(image, device):
    # Ensure the image is on the CPU
    image = image.cpu()

    # Create a transformation pipeline that resizes and normalizes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
    ])

    # Apply the transformation
    resized_and_normalized_image = transform(image)

    # Move the transformed image back to the appropriate device
    return resized_and_normalized_image.to(device)

def add_noise_to_image_like_diffusion(images, labels=None):
    P_std = 1.2
    P_mean = -1.2 - 0.5
    images_np = images.numpy()
    rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
    sigma = (rnd_normal * P_std + P_mean).exp()

    # add to background a certain gray value
    # Generate random gray levels for each image in the batch
    random_gray_level = torch.randint(0, 9, (images.shape[0], 1, 1, 1), device=images.device).float()
    gray_level = torch.randint(0, 2, (sigma.shape[0], 1, 32, 32), device=images.device) + random_gray_level.repeat(1, 1, 32, 32)
    images_np[images_np == 0] = gray_level[images_np == 0] / 255

    # 20% of the sigma are set to 0
    random_number_between_0_and_100 = torch.randint(0, 100, (sigma.shape[0], 1, 1, 1), device=images.device)
    sigma[random_number_between_0_and_100 < 20] = 0

    n = torch.randn_like(images) * sigma
    images_noisy = (images + n)

    if labels is not None:
        threshold_of_noise_class = 0.65
        labels_10 = torch.full_like(labels, 0)
        labels_10[:,:,-1] = labels_10[:,:,-1] + 1
        mask = sigma > threshold_of_noise_class
        mask = mask.squeeze(-1).squeeze(-1)
        labels[mask] = labels_10[mask]

    #plot_batch_of_image_and_noise(images, n, images_noisy, sigma)
    return images_noisy, labels

class ResNetForMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetForMNIST, self).__init__()
        self.num_classes = num_classes
        # Load a pre-trained ResNet18 model
        self.resnet = resnet18(pretrained=False)

        # Modify the first convolution layer to accept 1-channel input instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the last fully connected layer for 10 output classes (for digits 0-9)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Extract features before the fully connected layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)  # Flatten the features

        # Pass the features to the fully connected layer for classification
        output = self.resnet.fc(features)

        # Return both features and output
        return features, output

    # make the attribute number of classes
    def num_classes(self):
        return self.num_classes


# Training loop
def train(model, dataset_iterator, criterion, optimizer, device, transform, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        #dataset_iterator = iter(torch.utils.data.DataLoader(dataset=train_obj,sampler=dataset_sampler,batch_size=32,))
        num_iterations = len(dataset_iterator)
        t = trange(int(num_iterations), desc=f"Epoch {int(epoch) + 1}/{int(num_epochs)}", leave=False)
        for iteration in t:
            images, labels, frame_idx_dir_change = next(dataset_iterator)
            # align the dimension of the labels to the number of classes in the model
            labels = torch.cat((labels, torch.zeros(labels.size(0), 1, 1)), dim=2) if model.num_classes == 11 else labels
            images, labels = add_noise_to_image_like_diffusion(images.squeeze(-1), labels)
            images = transform_and_resize_on_cpu(images, device)

            # Zero the parameter gradients
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            _,outputs = model(images)
            loss = criterion(outputs, labels.squeeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataset_iterator):.4f}")


# Testing the model
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    num_iterations = 100

    with torch.no_grad():
        for iteration in range(num_iterations):
            images, labels, frame_idx_dir_change = next(test_loader)
            labels = torch.cat((labels, torch.zeros(labels.size(0), 1, 1)),
                               dim=2) if model.num_classes == 11 else labels
            images, labels = add_noise_to_image_like_diffusion(images.squeeze(-1), labels)
            images = transform_and_resize_on_cpu(images.squeeze(-1), device)

            labels = labels.squeeze(1).to(device)
            _,outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, true_label = torch.max(labels.data, 1)
            total += true_label.size(0)
            correct += (predicted == true_label).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')


def get_prediction(image, device, path, num_classes=10):
    # load the model pretrained model from path
    model = ResNetForMNIST(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
    ])

    image = transform(image)
    model.eval()
    image = image.to(device).float()
    features, output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted, features



if __name__ == '__main__':
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5 for grayscale images
    ])

    # Check if GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize the model, move it to the device
    model = ResNetForMNIST(num_classes=11).to(device)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5 for grayscale images
    ])

    moving_mnist_path = '/cephyr/users/julianjo/Alvis/data' if torch.cuda.is_available() else '/Users/juliankleutgens/data_edm/data/MNIST'
    dataset_obj = MovingMNIST(train=True, data_root=moving_mnist_path, seq_len=1, num_digits=1, image_size=32,
                              mode='middel',
                              deterministic=False, log_direction_change=True, step_length=0.1,
                              let_last_frame_after_change=False, use_label=True)
    dataset_sampler = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        shuffle=True,   # Shuffle the data during training
        )
    batchsize = 512
    num_epochs = 20
    want_to_test = False
    if not want_to_test:
        for i in range(num_epochs):
            dataset_iterator = iter(torch.utils.data.DataLoader(
                dataset=dataset_obj,
                sampler=dataset_sampler,
                batch_size=batchsize,
            ))

            # Train the model
            train(model, dataset_iterator, criterion, optimizer, device, transform, num_epochs=1)
    # Test the model
    if want_to_test:
        # load a model from the path /Users/juliankleutgens/PycharmProjects/edm-main/mnist_resnet18_5e.pth
        model.load_state_dict(torch.load('/Users/juliankleutgens/PycharmProjects/edm-main/mnist_resnet18_5e.pth'))
    dataset_iterator = iter(torch.utils.data.DataLoader(
        dataset=dataset_obj,
        sampler=dataset_sampler,
        batch_size=batchsize,
    ))
    test(model, dataset_iterator, device)

    # Save the trained model
    if not want_to_test:
        torch.save(model.state_dict(), 'mnist_resnet18.pth')
