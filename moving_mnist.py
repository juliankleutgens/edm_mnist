import socket
import numpy as np
import torch
from torchvision import datasets, transforms

class MovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True, digit_filter=None, move_horizontally=False, use_label=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 16
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.digit_filter = digit_filter
        self.channels = 1
        self.move_horizontally = move_horizontally
        self.use_label = use_label

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((self.digit_size, self.digit_size)),  # Resize to digit_size x digit_size
                 transforms.ToTensor()]) )

        #transform=transforms.Compose([transforms.Scale(self.digit_size),transforms.ToTensor()]))
        if self.digit_filter is not None:
            indices = []
            for i in range(len(self.data)):
                t = self.data.targets[i].item()
                if t in self.digit_filter:
                    indices.append(i)
            self.data = torch.utils.data.Subset(self.data, indices)


        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, label = self.data[idx]
            # If move_horizontally is True, apply the middle-crossing behavior
            if self.move_horizontally:
                # Starting position in the middle of the frame (both horizontally and vertically)
                sy = (image_size - digit_size) // 2  # Fixed vertical position at the center
                sx = np.random.randint(image_size - digit_size)  # Random horizontal starting point
                direction = np.random.choice([-1, 1])  # Randomly choose left (-1) or right (1)

                for t in range(self.seq_len):
                    # Update the horizontal position
                    sx += direction * 2  # Move in the chosen direction

                    # Check if the digit crosses the middle
                    if (sx + digit_size // 2) == image_size // 2:
                        # Randomly choose a new direction when crossing the middle
                        direction = np.random.choice([-1, 1])

                    # Handle boundary conditions for horizontal movement
                    if sx < 0:
                        sx = 0
                        direction = 1  # Force movement to the right when hitting the left boundary
                    elif sx >= image_size - digit_size:
                        sx = image_size - digit_size - 1
                        direction = -1  # Force movement to the left when hitting the right boundary

                    # Only sx (horizontal) is updated; sy (vertical) remains constant
                    x[t, sy:sy + digit_size, sx:sx + digit_size, 0] += digit.numpy().squeeze()

            # Otherwise, use the original free movement behavior
            else:

                sx = np.random.randint(image_size-digit_size)
                sy = np.random.randint(image_size-digit_size)
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
                for t in range(self.seq_len):
                    if sy < 0:
                        sy = 0
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(1, 5)
                            dx = np.random.randint(-4, 5)
                    elif sy >= image_size-self.digit_size:
                        sy = image_size-self.digit_size-1
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(-4, 0)
                            dx = np.random.randint(-4, 5)

                    if sx < 0:
                        sx = 0
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(1, 5)
                            dy = np.random.randint(-4, 5)
                    elif sx >= image_size-self.digit_size:
                        sx = image_size-self.digit_size-1
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(-4, 0)
                            dy = np.random.randint(-4, 5)

                    x[t, sy:sy+digit_size, sx:sx+digit_size, 0] += digit.numpy().squeeze()
                    sy += dy
                    sx += dx

        x[x>1] = 1.
        if self.use_label:
            labels = np.array([label]*self.seq_len)
            return x, self.make_onehot_label(torch.tensor(labels))
        return x, np.empty((1,0))

    def close(self):
        pass  # No resources to clean up

    def _load_raw_labels(self):
        """Load and return the raw labels (if any)."""
        return None  # No labels in MovingMNIST

    def name(self):
        return 'MovingMNIST'

    @property
    def image_shape(self):
        return [self.channels, self.image_size, self.image_size]

    @property
    def num_channels(self):
        return self.channels

    @property
    def resolution(self):
        return self.image_size

    @property
    def label_shape(self):
        return [0]  # No labels in MovingMNIST

    @property
    def label_dim(self):
        if self.use_label:
            return 10
        return 0  # No label dimension

    @property
    def has_labels(self):
        return False

    @property
    def has_onehot_labels(self):
        return False

    def make_onehot_label(self, labels_tensor):
        onehot_labels = torch.zeros(labels_tensor.size(0), 10)
        onehot_labels.scatter_(1, labels_tensor.unsqueeze(1), 1)
        return onehot_labels


import matplotlib.pyplot as plt


def main():
    # Define the parameters for the MovingMNIST dataset
    train = True  # Set to False if you want to use the test set
    data_root = './data'  # The directory where the MNIST data will be downloaded
    seq_len = 64  # Length of the sequence
    num_digits = 1  # Number of digits to display in each sequence
    image_size = 32  # Size of the image frame
    deterministic = True  # Whether the movement should be deterministic
    digit_filter = [8]

    # Initialize the MovingMNIST dataset
    moving_mnist = MovingMNIST(train, data_root, seq_len, num_digits, image_size, deterministic, digit_filter=digit_filter, move_horizontally=True)

    # Sample an item from the dataset
    sample_index = 0  # Index of the sample to visualize
    sample, label = moving_mnist[sample_index]

    # Visualize the sampled sequence
    fig, axes = plt.subplots(1, seq_len, figsize=(seq_len, 2))
    for i in range(seq_len):
        axes[i].imshow(sample[i, :, :, 0], cmap='gray')
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__":
    main()
