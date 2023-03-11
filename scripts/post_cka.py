# --------------------------------------------------------------------------------
# CKA Visualization.
#
# Modified by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tqdm import tqdm


def unbiased_hsic(matrix_k, matrix_l):
    r"""Computes an unbiased estimator of HSIC.

    HSIC is short for the Hilbert-Schmid-Independence Criterion, which measures
    the statistical independence between two distributions. In our case, these
    distributions are the activations.

    This is equation (2) from the paper "Do Wide And Deep Networks Learn The
    Same Things?".

    Modified from
    https://phrasenmaeher.io/do-different-neural-networks-learn-the-same-things.

    """

    # create the unit **vector** filled with ones
    n = matrix_k.shape[0]
    ones = np.ones(shape=n)

    # fill the diagonal entries with zeros
    np.fill_diagonal(matrix_k, val=0)  # this is now K_tilde
    np.fill_diagonal(matrix_l, val=0)  # this is now L_tilde

    # first part in the square brackets
    trace = np.trace(np.dot(matrix_k, matrix_l))

    # middle part in the square brackets
    nominator1 = np.dot(np.dot(ones.T, matrix_k), ones)
    nominator2 = np.dot(np.dot(ones.T, matrix_l), ones)
    denominator = (n - 1) * (n - 2)
    middle = np.dot(nominator1, nominator2) / denominator

    # third part in the square brackets
    multiplier1 = 2 / (n - 2)
    multiplier2 = np.dot(np.dot(ones.T, matrix_k),
                         np.dot(matrix_l, ones))
    last = multiplier1 * multiplier2

    # complete equation
    return 1 / (n * (n - 3)) * (trace + middle - last)


def cal_cka(matrix_x, matrix_y):
    r"""Computes the CKA of two matrices.

    This is equation (1) from the paper "Do Wide And Deep Networks Learn The
    Same Things?".

    Modified from
    https://phrasenmaeher.io/do-different-neural-networks-learn-the-same-things.

    """

    nominator = unbiased_hsic(np.dot(matrix_x, matrix_x.T),
                              np.dot(matrix_y, matrix_y.T))
    denominator1 = unbiased_hsic(np.dot(matrix_x, matrix_x.T),
                                 np.dot(matrix_x, matrix_x.T))
    denominator2 = unbiased_hsic(np.dot(matrix_y, matrix_y.T),
                                 np.dot(matrix_y, matrix_y.T))

    return nominator / np.sqrt(denominator1 * denominator2)


def cal_similarity(act_a, act_b):
    r"""Takes two activations A and B and computes the linear CKA to measure
        their similarity.

    Modified from
    https://phrasenmaeher.io/do-different-neural-networks-learn-the-same-things.

    """

    # unfold the activations, that is make a (n, h*w*c) representation
    shape = act_a.shape
    act_a = np.reshape(act_a, newshape=(shape[0], np.prod(shape[1:])))

    shape = act_b.shape
    act_b = np.reshape(act_b, newshape=(shape[0], np.prod(shape[1:])))

    # calculate the CKA score
    cka_score = cal_cka(act_a, act_b)

    del act_a
    del act_b

    return cka_score


def compare_activations(act1, act2):
    r"""Calculate pairwise comparison of hidden representations.
    """

    # create a placeholder array
    result_array = np.zeros(shape=(len(act1), len(act2)))

    i = 0
    for _act1 in tqdm(act1):
        j = 0
        for _act2 in act2:
            cka_score = cal_similarity(_act1, _act2)
            result_array[i, j] = cka_score
            j += 1
        i += 1

    return result_array


def plot_cka(pkl_path1, pkl_path2=None):
    with open(pkl_path1, 'rb') as f:
        pkl1 = pickle.load(f)

    if pkl_path2 is not None:
        with open(pkl_path2, 'rb') as f:
            pkl2 = pickle.load(f)
    else:
        pkl2 = pkl1

    similarity_matrix = compare_activations(pkl1, pkl2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    im = ax.imshow(similarity_matrix, cmap='magma', vmin=0.0, vmax=1.0)
    ax.axes.invert_yaxis()
    plt.colorbar(im)
    plt.savefig('cka_result.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'i',
        type=str,
        default=None,
        help=''
    )
    parser.add_argument(
        '--j',
        type=str,
        default=None,
        help=''
    )
    args = parser.parse_args()

    plot_cka(args.i, args.j)
