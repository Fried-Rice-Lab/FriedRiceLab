import pickle

import numpy as np
from matplotlib import pyplot as plt


def compute_distance_matrix(num_pixels):
    r"""Helper function to compute distance matrix.
    """

    length = int(np.sqrt(num_pixels))

    distance_matrix = np.zeros((num_pixels, num_pixels))
    for i in range(num_pixels):
        for j in range(num_pixels):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def cal_mean_distance_fast(pkl_path, window_list: tuple = ((12, 12), (12, 12))):
    # get attention_weights
    with open(pkl_path, 'rb') as f:
        # shape: num_images num_layers num_groups num_heads num_patches num_pixels num_pixels
        # num_images: Number of images used
        # num_layers: Number of layers hooked
        # num_groups: Number of SA groups
        # num_heads: Number of SA heads
        # num_patches: Number of patches in an image. Suppose the image size is (H, W) and
        #     the patch size is (a, b), then num_patches = HW/ab
        # num_pixels: Number of pixels in a patch
        attention_weights = pickle.load(f)

    # print(len(attention_weights),
    #       len(attention_weights[0]),
    #       len(attention_weights[0][0]),
    #       len(attention_weights[0][0][0]),
    #       attention_weights[0][0][0].shape)

    # average across all patches in an image (reduce memory usage)
    # shape: num_images num_layers num_groups num_heads num_pixels num_pixels
    attention_weights = np.mean(attention_weights, axis=-3)

    # now average across all heads in a layer (reduce memory usage)
    # shape: num_images num_layers num_groups num_pixels num_pixels
    attention_weights = np.mean(attention_weights, axis=-3)

    # shape: num_groups num_images num_layers num_pixels num_pixels
    attention_weights = np.transpose(attention_weights, (2, 0, 1, 3, 4))

    assert len(attention_weights) == len(window_list), \
        f"Attention weights have {len(attention_weights)} groups, but only " \
        f"{len(window_list)} window sizes are given."

    mean_distances = 0
    for group_weights, window_size in zip(attention_weights, window_list):
        assert window_size[0] == window_size[1], \
            "Only square window size is supported now."

        # get distance matrix
        distance_matrix = compute_distance_matrix(window_size[0] * window_size[1])
        distance_matrix = distance_matrix[np.newaxis, np.newaxis, :]

        # make attention_weights weighted by distance matrix
        # shape: num_images num_layers num_pixels num_pixels
        group_distances = group_weights * distance_matrix

        # sum along last axis to get average distance per pixel
        # this is due to the fact that they are softmaxed
        # shape: num_images num_layers num_pixels
        group_distances = np.sum(group_distances, axis=-1)

        # now average across all pixels in a patch
        # shape: num_images num_layers
        group_distances = np.mean(group_distances, axis=-1)
        mean_distances += group_distances
    mean_distances /= len(attention_weights)

    # shape: num_layers num_images
    return np.transpose(mean_distances, (1, 0))


def plot_mad(pkl_path):
    md = cal_mean_distance_fast(pkl_path)

    md_mean = np.mean(md, axis=-1)
    md_std = np.std(md, axis=-1)

    x = np.arange(0, md.shape[0])

    plt.plot(x, md_mean)
    plt.fill_between(x, md_mean - md_std, md_mean + md_std,
                     color='blue', alpha=0.2)

    plt.savefig('mad_result.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'i',
        type=str,
        default=None,
        help=''
    )
    args = parser.parse_args()

    plot_mad(args.i)
