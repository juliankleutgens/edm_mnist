
def convert_video2imges_in_batch(images):
    # images: (N, T, H, W, C)
    N, T, H, W, C = images.size()
    images = images.view(N*T, C, H, W)
    return images