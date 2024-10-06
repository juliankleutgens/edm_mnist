import socket
import numpy as np
import torch
from torchvision import datasets, transforms

class MovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True, digit_filter=None, mode='free',
                 use_label=False, step_length=0.1, log_direction_change=False, prob_direction_change=0, let_last_frame_after_change=True
                 ,num_of_directions_in_circle=8):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = step_length
        self.digit_size = 16
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.digit_filter = digit_filter
        self.channels = 1
        self.mode = mode # there are three modes: free, horizontal, and circle
        self.use_label = use_label
        self.direction_change = log_direction_change
        self.p = prob_direction_change # Probability of changing direction
        if let_last_frame_after_change and mode == 'horizontal':
            self.p = 1.0 - self.p # the sequence get mirrored in the end
        if mode == 'horizontal' and self.p == 0:
            self.p = 0.5
            print("Attention: prob_direction_change is set to 0.5 for horizontal mode")

        self.let_last_frame_after_change = let_last_frame_after_change
        if num_of_directions_in_circle not in [4, 8]:
            raise ValueError("num_of_directions_in_circle must be 4 or 8")
        self.num_of_directions = num_of_directions_in_circle

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

        frame_idx_dir_change = []

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, label = self.data[idx]
            # If move_horizontally is True, apply the middle-crossing behavior
            if 'horizontal' in self.mode:
                x, frame_idx_dir_change = self._move_horizontally(digit, x, frame_idx_dir_change)
            elif 'circle' in self.mode:
                x, frame_idx_dir_change = self.jump_randomly_in_eight_directions_back_and_forth(digit, x, frame_idx_dir_change)
            elif  'free' in self.mode:
                x, frame_idx_dir_change = self._move_free(digit, x, frame_idx_dir_change)
            else:
                raise ValueError("mode must be one of 'horizontal', 'circle', or 'free'")

        x[x>1] = 1.
        # fill the list frame_idx_dir_change with zeros to match the length of the sequence
        frame_idx_dir_change = torch.tensor(frame_idx_dir_change)
        if self.let_last_frame_after_change:
            x = np.flip(x,axis=0).copy() # mirror the sequence
            frame_idx_dir_change = (self.seq_len - 1) - frame_idx_dir_change
        # Concatenate frame_idx_dir_change with -1 filled tensor to match the sequence length
        frame_idx_dir_change = torch.cat((frame_idx_dir_change, torch.ones(self.seq_len - len(frame_idx_dir_change)) * -1),dim=0)

        if self.use_label:
            labels = np.array([label]*self.seq_len)
            if self.direction_change:
                return x, self.make_onehot_label(torch.tensor(labels)), frame_idx_dir_change
            return x, self.make_onehot_label(torch.tensor(labels))
        if self.direction_change:
            return x, np.empty((1,0)), frame_idx_dir_change
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

    def _move_free(self, digit, x, frame_idx_dir_change):
        image_size = self.image_size
        digit_size = self.digit_size
        sx = np.random.randint(image_size - digit_size)
        sy = np.random.randint(image_size - digit_size)
        dx = np.random.randint(-4, 5)
        dy = np.random.randint(-4, 5)

        for t in range(self.seq_len):
            if sy < 0:
                sy = 0
                dy = -dy if self.deterministic else np.random.randint(1, 5)
                dx = np.random.randint(-4, 5)
                frame_idx_dir_change.append(t)
            elif sy >= image_size - digit_size:
                sy = image_size - digit_size - 1
                dy = -dy if self.deterministic else np.random.randint(-4, 0)
                dx = np.random.randint(-4, 5)
                frame_idx_dir_change.append(t)

            if sx < 0:
                sx = 0
                dx = -dx if self.deterministic else np.random.randint(1, 5)
                dy = np.random.randint(-4, 5)
                frame_idx_dir_change.append(t)
            elif sx >= image_size - digit_size:
                sx = image_size - digit_size - 1
                dx = -dx if self.deterministic else np.random.randint(-4, 0)
                dy = np.random.randint(-4, 5)
                frame_idx_dir_change.append(t)

            x[t, sy:sy + digit_size, sx:sx + digit_size, 0] += digit.numpy().squeeze()
            sy += dy
            sx += dx

        return x, frame_idx_dir_change

    def _move_horizontally(self, digit, x, frame_idx_dir_change):
        image_size = self.image_size
        digit_size = self.digit_size
        sy = (image_size - digit_size) // 2
        sx = np.random.randint(image_size - digit_size) if not self.let_last_frame_after_change else image_size // 2 - digit_size // 2 + 4 * (-np.random.choice([-1, 1], p=[1 - self.p, self.p]))
        direction = np.random.choice([-1, 1], p=[1 - self.p, self.p])
        change_direction_last_frame = False

        for t in range(self.seq_len):
            sx_prev = sx
            sx += direction * 2

            if ((sx_prev + digit_size // 2 < image_size // 2 and sx + digit_size // 2 >= image_size // 2) or
                (sx_prev + digit_size // 2 > image_size // 2 and sx + digit_size // 2 <= image_size // 2)) and not change_direction_last_frame:
                direction = np.random.choice([-1, 1], p=[1 - self.p, self.p])
                frame_idx_dir_change.append(t)
                change_direction_last_frame = True
            else:
                change_direction_last_frame = False

            if sx < 0:
                sx = 0
                direction = 1
            elif sx >= image_size - digit_size:
                sx = image_size - digit_size - 1
                direction = -1

            x[t, sy:sy + digit_size, sx:sx + digit_size, 0] += digit.numpy().squeeze()

        return x, frame_idx_dir_change

    def jump_randomly_in_eight_directions_back_and_forth(self, digit, x, frame_idx_dir_change):
        """Make the digit jump in 8 directions clockwise, then back to the center."""
        image_size = self.image_size
        digit_size = self.digit_size

        cx = (image_size - digit_size) // 2
        cy = (image_size - digit_size) // 2
        step = 4

        # 8 directions in clockwise order
        if self.num_of_directions == 8:
            directions = np.array([
                [-1, 0] ,   # Up
                [-1, 1],   # Up-right
                [0, 1],    # Right
                [1, 1],    # Down-right
                [1, 0],    # Down
                [1, -1],   # Down-left
                [0, -1],   # Left
                [-1, -1]   # Up-left
            ])
        elif self.num_of_directions == 4:
            directions = np.array([
                [-1, 0],  # Up
                [0, 1],   # Right
                [1, 0],   # Down
                [0, -1]   # Left
            ])
        elif self.num_of_directions == 2:
            directions = np.array([
                [0, 1],   # Right#
                [0, -1]   # Left
            ])

        for t in range(self.seq_len):
            x[t, cy:cy + digit_size, cx:cx + digit_size, 0] += digit.numpy().squeeze()
            if t % 2 == 0:
                if self.p == 0:
                    random_direction = np.random.randint(self.num_of_directions)
                else:
                    random_direction = np.random.choice(self.num_of_directions, p=[self.p] + [(1 - self.p)/(self.num_of_directions-1)] * (self.num_of_directions - 1))

                direction = directions[random_direction]
                dx = direction[0] * step
                dy = direction[1] * step
                frame_idx_dir_change.append(t)
            else:
                dx = -direction[0] * step
                dy = -direction[1] * step
            cx += dx
            cy += dy
        return x, frame_idx_dir_change

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
    seq_len = 10  # Length of the sequence
    num_digits = 1  # Number of digits to display in each sequence
    image_size = 32  # Size of the image frame
    deterministic = False  # Whether the movement should be deterministic
    digit_filter = [8]

    # Initialize the MovingMNIST dataset
    moving_mnist = MovingMNIST(train, data_root, seq_len, num_digits, image_size, deterministic, digit_filter=digit_filter, mode='circle'
                                 , use_label=True, log_direction_change=True, prob_direction_change=0.3, let_last_frame_after_change=False)

    # Sample an item from the dataset
    sample_index = 0  # Index of the sample to visualize
    sample, label, _ = moving_mnist[sample_index]

    # Visualize the sampled sequence
    fig, axes = plt.subplots(1, seq_len, figsize=(seq_len, 2))
    for i in range(seq_len):
        axes[i].imshow(sample[i, :, :, 0], cmap='gray')
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__":
    main()
