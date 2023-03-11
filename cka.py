# --------------------------------------------------------------------------------
# Helper function for CKA Visualization.
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
        batch_size=len(dataset),
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


def cka_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"cka_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create cka dataset and dataloader
    cka_loaders = []
    for _, dataset_opt in sorted(opt['cka_datasets'].items()):
        dataset_opt['phase'] = 'val'
        dataset_opt['bit'] = opt['bit']
        dataset_opt['scale'] = opt['scale']
        cka_set = build_dataset(dataset_opt)
        cka_loader = build_dataloader(cka_set, dataset_opt)
        logger.info(f"Number of cka images in {dataset_opt['name']}: {len(cka_set)}")
        cka_loaders.append(cka_loader)

    # create model
    model = build_model(opt)

    # import hook layer
    module = importlib.import_module('archs.utils')
    hook_layer_type = getattr(module, opt['hook_layer_type'])

    for cka_loader in cka_loaders:
        cka_set_name = cka_loader.dataset.opt['name']
        logger.info(f'Ckaing {cka_set_name}...')
        cka_outputs = model.nondist_cka(cka_loader, hook_layer_type)
        pkl_path = os.path.join(opt['path']['results_root'], f"{opt['name']}_{cka_set_name}_cka.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(cka_outputs, f)

    logger.info(f'End of ckaing.')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    cka_pipeline(root_path)
