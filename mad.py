# --------------------------------------------------------------------------------
# Helper function for MAD Visualization.
#
# Implemented by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
import importlib
import logging
import os.path
import pickle
from os import path as osp

import torch
import torch.utils.data
from basicsr.data import build_dataset
from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str
from basicsr.utils.options import dict2str

import archs  # noqa
import data  # noqa
import models  # noqa
from utils import parse_options, make_exp_dirs


def build_dataloader(dataset, dataset_opt):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options.
    """

    dataloader_args = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=None,
        drop_last=True,
        worker_init_fn=None,
        pin_memory=False,
        persistent_workers=False
    )

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def mad_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"mad_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create mad dataset and dataloader
    mad_loaders = []
    for _, dataset_opt in sorted(opt['mad_datasets'].items()):
        dataset_opt['phase'] = 'val'
        dataset_opt['bit'] = opt['bit']
        dataset_opt['scale'] = opt['scale']
        mad_set = build_dataset(dataset_opt)
        mad_loader = build_dataloader(mad_set, dataset_opt)
        logger.info(f"Number of mad images in {dataset_opt['name']}: {len(mad_set)}")
        mad_loaders.append(mad_loader)

    # create model
    model = build_model(opt)

    # import hook layer
    module = importlib.import_module('archs.utils')
    hook_layer_type = getattr(module, opt['hook_layer_type'])

    for mad_loader in mad_loaders:
        mad_set_name = mad_loader.dataset.opt['name']
        logger.info(f'Mading {mad_set_name}...')
        mad_outputs = model.nondist_mad(mad_loader, hook_layer_type)
        pkl_path = os.path.join(opt['path']['results_root'],
                                f"{opt['name']}_{mad_set_name}_mad.pkl")
        with open(pkl_path, 'wb') as f:
            # shape: num_images num_layers num_groups num_heads num_patches num_pixels num_pixels
            # num_images: Number of images used
            # num_layers: Number of layers hooked
            # num_groups: Number of SA groups
            # num_heads: Number of SA heads
            # num_patches: Number of patches in an image. Suppose the image size is (H, W) and
            #     the patch size is (a, b), then num_patches = HW/ab
            # num_pixels: Number of pixels in a patch
            pickle.dump(mad_outputs, f)

    logger.info(f'End of mading.')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    mad_pipeline(root_path)
