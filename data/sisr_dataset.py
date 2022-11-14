# --------------------------------------------------------------------------------
# Basic SR Dataset. Supports pixel value between [0, 1] or [0, 255].
#
# Implemented by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
import imageio.v2 as imageio
import numpy as np
import torch
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SISRDataset(PairedImageDataset):
    r"""Paired image dataset for image restoration.
        Pixel values are between [0., 1.].
    """

    def __init__(self, opt) -> None:
        super(SISRDataset, self).__init__(opt)


@DATASET_REGISTRY.register()
class SISRDataset8Bit(PairedImageDataset):
    r"""Paired image dataset for image restoration.
        Pixel values are between [0., 255.].
    """

    def __init__(self, opt) -> None:
        super(SISRDataset8Bit, self).__init__(opt)

    def __getitem__(self, index) -> dict:
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

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

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_gt, img_lq = self.np2tensor([img_gt, img_lq])

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def np2tensor(imgs: list) -> list:
        def _np2tensor(img):
            return torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float()

        return [_np2tensor(img) for img in imgs]
