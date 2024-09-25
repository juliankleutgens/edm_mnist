
def convert_video2images_in_batch(images, labels, use_label=False):
    # images: (N, T, H, W, C)
    N, T, H, W, C = images.size()
    images = images.view(N*T, C, H, W)

    if use_label:
        labels = labels.view(N*T, 10)
        return images, labels
    return images, labels