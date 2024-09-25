import torch
def convert_video2images_in_batch(images, labels, use_label=False, num_cond_frames=0):
    if num_cond_frames==0:
        # images: (N, T, H, W, C)
        N, T, H, W, C = images.size()
        images = images.view(N*T, C, H, W)

        if use_label:
            labels = labels.view(N*T, 10)
            return images, labels
        return images, labels
    else:
        N, T, H, W, C = images.size()
        img_buffer = torch.ones(N*(T-num_cond_frames), C+num_cond_frames, H, W)
        for i in range(0,T-num_cond_frames):
            img_buffer[(N*i):(N*(i+1)),:,:,:] = images[:,i:i+num_cond_frames+1,:,:,:].view(N, C+num_cond_frames, H, W)
        #polt_image_sequence(img_buffer)
        if use_label:
            labels = labels.view(N*T, 10)
            return img_buffer, labels
        return img_buffer, labels



def polt_image_sequence(images):
    from matplotlib import pyplot as plt
    b, images_per_sequence, _, _ = images.size()  # Assuming `images` has shape [b, num_images, H, W]
    num_sequences = 8  # The number of sequences you want to display


    fig, ax = plt.subplots(num_sequences, images_per_sequence,
                           figsize=(15, 10))  # Create a grid of 8 rows and 3 columns

    for seq in range(num_sequences):
        for img in range(images_per_sequence):
            ax[seq, img].imshow(images[seq, img, :, :].squeeze(), cmap='gray')  # Display each image from the sequence
            ax[seq, img].axis('off')  # Remove axis

    plt.tight_layout()
    plt.show()

