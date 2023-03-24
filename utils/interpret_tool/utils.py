import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
import torchvision
from PIL import Image
from scipy import stats

IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm']


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def reduce_func(method):
    """

    :param method: ['mean', 'sum', 'max', 'min', 'count', 'std']
    :return:
    """
    if method == 'sum':
        return torch.sum
    elif method == 'mean':
        return torch.mean
    elif method == 'count':
        return lambda x: sum(x.size())
    else:
        raise NotImplementedError()


def attr_id(tensor, h, w, window=8, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    crop = tensor[:, :, h: h + window, w: w + window]
    return reduce_func(reduce)(crop)


def attr_grad(tensor, h, w, window=8, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2)
    w_grad = torch.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = torch.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    crop = grad[:, :, h: h + window, w: w + window]
    return reduce_func(reduce)(crop)


# gabor_filter = cv2.getGaborKernel((21, 21), 10.0, -np.pi/4, 8.0, 1, 0, ktype=cv2.CV_32F)

def attr_gabor_generator(gabor_filter):
    filter = torch.from_numpy(gabor_filter).view((1, 1,) + gabor_filter.shape).repeat(1, 3, 1, 1)

    def attr_gabor(tensor, h, w, window=8, reduce='sum'):
        after_filter = f.conv2d(tensor, filter, bias=None)
        crop = after_filter[:, :, h: h + window, w: w + window]
        return reduce_func(reduce)(crop)

    return attr_gabor


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pil_loader(path, mode='RGB'):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: PIL.Image
    """
    assert _is_image_file(path), "%s is not an image" % path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def calculate_RF(model):
    layers = getLayers(model)
    r = 1
    for layer in layers[::-1]:
        if isinstance(layer, torch.nn.Conv2d):
            kernel = layer.kernel_size[0]
            padding = layer.padding[0]
            stride = layer.stride[0]
            r = stride * r + (kernel - stride)
    return r


def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []

    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


def load_as_tensor(path, mode='RGB'):
    """
    Load image to tensor
    :param path: image path
    :param mode: 'Y' returns 1 channel tensor, 'RGB' returns 3 channels, 'RGBA' returns 4 channels, 'YCbCr' returns 3 channels
    :return: 3D tensor
    """
    if mode != 'Y':
        return PIL2Tensor(pil_loader(path, mode=mode))
    else:
        return PIL2Tensor(pil_loader(path, mode='YCbCr'))[:1]


def PIL2Tensor(pil_image):
    return torchvision.transforms.functional.to_tensor(pil_image)


def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return torchvision.transforms.functional.to_pil_image(tensor_image.detach(), mode=mode)


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def image_files(path):
    """
    return list of images in the path
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = os.listdir(abs_path)
    for i in range(len(image_files)):
        if (not os.path.isdir(image_files[i])) and (_is_image_file(image_files[i])):
            image_files[i] = os.path.join(abs_path, image_files[i])
    return image_files


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _sigmoid_to_tanh(x):
    """
    range [0, 1] to range [-1, 1]
    :param x: tensor type
    :return: tensor
    """
    return (x - 0.5) * 2.


def _tanh_to_sigmoid(x):
    """
    range [-1, 1] to range [0, 1]
    :param x:
    :return:
    """
    return x * 0.5 + 0.5


def _add_batch_one(tensor):
    """
    Return a tensor with size (1, ) + tensor.size
    :param tensor: 2D or 3D tensor
    :return: 3D or 4D tensor
    """
    return tensor.view((1,) + tensor.size())


def _remove_batch(tensor):
    """
    Return a tensor with size tensor.size()[1:]
    :param tensor: 3D or 4D tensor
    :return: 2D or 3D tensor
    """
    return tensor.view(tensor.size()[1:])


def mod_crop(tensor, scale=4):
    B, C, H, W = tensor.shape
    return tensor[:, :, :H - H % scale, :W - W % scale]


def cv2_to_pil(img):
    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def make_pil_grid(pil_image_list):
    sizex, sizey = pil_image_list[0].size
    for img in pil_image_list:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    target = Image.new('RGB', (sizex * len(pil_image_list), sizey))
    left = 0
    right = sizex
    for i in range(len(pil_image_list)):
        target.paste(pil_image_list[i], (left, 0, right, sizey))
        left += sizex
        right += sizex
    return target


def blend_input(map, input):
    return Image.blend(map, input, 0.4)


def count_saliency_pixels(map, threshold=0.95):
    sum_threshold = map.reshape(-1).sum() * threshold
    cum_sum = -np.cumsum(np.sort(-map.reshape(-1)))
    return len(cum_sum[cum_sum < sum_threshold])


def plot_diff_of_attrs_kde(A, B, zoomin=4, blend=0.5):
    grad_flat = A.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:A.shape[0]:1, 0:A.shape[1]:1]
    Y, X = np.mgrid[0:A.shape[0]:1, 0:A.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Za = np.reshape(kernel(positions).T, A.shape)
    Za = Za / Za.max()

    grad_flat = B.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:B.shape[0]:1, 0:B.shape[1]:1]
    Y, X = np.mgrid[0:B.shape[0]:1, 0:B.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Zb = np.reshape(kernel(positions).T, B.shape)
    Zb = Zb / Zb.max()

    diff = Za - Zb
    diff_norm = diff / diff.max()
    vis = Zb - blend * diff_norm

    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(vis * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)


def vis_saliency_kde(map, zoomin=4):
    grad_flat = map.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)


def vis_saliency(map, zoomin=4):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(map * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(map)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    Img = Img.resize((s1 * zoomin, s2 * zoomin), Image.NEAREST)
    return Img.convert('RGB')


def click_select_position(pil_img, window_size=16):
    """

    :param pil_img:
    :param window_size:
    :return: w, h
    """
    cv2_img = pil_to_cv2(pil_img)
    position = [-1, -1]

    def mouse(event, x, y, flags, param):
        """"""
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d, %d" % (x, y)
            position[0] = x
            position[1] = y
            draw_img = cv2_img.copy()
            cv2.rectangle(draw_img, (x, y), (x + window_size, y + window_size), (0, 0, 255), 2)
            cv2.putText(draw_img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", draw_img)

    cv2.namedWindow("image")
    cv2.imshow("image", cv2_img)
    cv2.resizeWindow("image", 800, 600)
    cv2.setMouseCallback("image", mouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return_img = cv2_img.copy()
    cv2.rectangle(return_img, (position[0], position[1]), (position[0] + window_size, position[1] + window_size),
                  (0, 0, 255), 2)
    return position[0], position[1], cv2_to_pil(return_img)


def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil


def grad_abs_norm(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad.sum(axis=0))
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max
    return grad_norm


def grad_norm(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = grad.sum(axis=0)
    grad_max = max(grad_2d.max(), abs(grad_2d.min()))
    grad_norm = grad_2d / grad_max
    return grad_norm


def grad_abs_norm_singlechannel(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad)
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max
    return grad_norm


def IG_baseline(numpy_image, mode='gaus'):
    """
    :param numpy_image: cv2 image
    :param mode:
    :return:
    """
    if mode == 'l1':
        raise NotImplementedError()
    elif mode == 'gaus':
        ablated = cv2.GaussianBlur(numpy_image, (7, 7), 0)
    elif mode == 'bif':
        ablated = cv2.bilateralFilter(numpy_image, 15, 90, 90)
    elif mode == 'mean':
        ablated = cv2.medianBlur(numpy_image, 5)
    else:
        ablated = cv2.GaussianBlur(numpy_image, (7, 7), 0)
    return ablated


def interpolation(x, x_prime, fold, mode='linear'):
    diff = x - x_prime
    l = np.linspace(0, 1, fold).reshape((fold, 1, 1, 1))
    interp_list = l * diff + x_prime
    return interp_list


def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    return kernel / np.sum(kernel)


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
