# --------------------------------------------------------------------------------
# Basic dataset for image restoration.
# Supports pixel value between [0, 2 ** Bit - 1].
#
# Implemented by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
import imageio.v2 as imageio
import numpy as np
import torch
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from .data_util import paired_paths_from_folder, paired_paths_from_lmdb, \
    paired_paths_from_meta_info_file


@DATASET_REGISTRY.register()
class IRDataset(data.Dataset):
    r"""Basic dataset for image restoration.
    """

    def __init__(self, opt) -> None:
        super(IRDataset, self).__init__()

        self.opt = opt
        self.bit = self.opt['bit']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.input_size = opt['input_size'] if 'input_size' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            if 'meta_info_file' in self.opt and \
                    self.opt['meta_info_file'] != 'None' and \
                    self.opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                    self.opt['meta_info_file'])
            else:
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and \
                self.opt['meta_info_file'] != 'None' and \
                self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index) -> dict:
        scale = self.opt['scale']
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if self.bit == 0:
            # Load gt and lq images. Dimension order: HWC; channel order: BGR;
            # image range: [0, 1], float32.
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

            # augmentation for training
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                # flip, rotation
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # color space transform
            if 'color' in self.opt and self.opt['color'] == 'y':
                img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
                img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

            # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
            # TODO: It is better to update the datasets, rather than force to crop
            if self.opt['phase'] != 'train':
                img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)
        else:
            # Load gt and lq images.
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = imageio.imread(img_bytes)

            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            # img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = imageio.imread(img_bytes)

            # augmentation for training
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.opt['scale'], gt_path)
                # flip, rotation
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
            # TODO: It is better to update the datasets, rather than force to crop
            if self.opt['phase'] != 'train':
                img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

            # BGR to RGB, HWC to CHW, numpy to tensor
            # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
            img_gt, img_lq = self.np2tensor([img_gt, img_lq])

        # Used in CKA and MAD to keep all inputs with the same shape
        if self.input_size is not None:
            # Cropping from the top right corner
            img_lq = img_lq[:, :self.input_size, :self.input_size]
            img_gt = img_gt[:, :self.input_size * scale, :self.input_size * scale]

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def np2tensor(imgs: list) -> list:
        def _np2tensor(img):
            return torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float()

        return [_np2tensor(img) for img in imgs]
