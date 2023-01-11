import logging
from os import path as osp

import torch
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str

import archs  # noqa
import data  # noqa
import models  # noqa
from utils import parse_options


def infer_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"infer_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create infer dataset and dataloader
    infer_loaders = []
    for _, dataset_opt in sorted(opt['infer_datasets'].items()):
        dataset_opt['phase'], dataset_opt['dataroot_lq'] = 'val', dataset_opt['dataroot_gt']  # fix it
        dataset_opt['bit'] = opt['bit']
        infer_set = build_dataset(dataset_opt)
        infer_loader = build_dataloader(
            infer_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of infer images in {dataset_opt['name']}: {len(infer_set)}")
        infer_loaders.append(infer_loader)

    # create model
    model = build_model(opt)

    for infer_loader in infer_loaders:
        infer_set_name = infer_loader.dataset.opt['name']
        logger.info(f'Inferring {infer_set_name}...')
        model.nondist_inference(infer_loader)

    logger.info(
        f"Inference ended. The results are saved to "
        f"{osp.join('results', opt['name'], 'visualization', 'inference')}.")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    infer_pipeline(root_path)
