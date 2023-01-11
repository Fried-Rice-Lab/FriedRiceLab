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
from tools.analyse_tool import get_model_flops, get_model_activation
from utils import parse_options


def analyse_pipeline(root_path, img_size: tuple = (3, 256, 256)):  # noqa
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False  # noqa

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"analyse_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create analyse dataset and dataloader
    analyse_loaders = []
    for _, dataset_opt in sorted(opt['analyse_datasets'].items()):
        dataset_opt['phase'] = 'val'
        dataset_opt['bit'] = opt['bit']
        analyse_set = build_dataset(dataset_opt)
        analyse_loader = build_dataloader(
            analyse_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of analyse images in {dataset_opt['name']}: {len(analyse_set)}")
        analyse_loaders.append(analyse_loader)

    # create model
    model = build_model(opt)

    logger.info(f'Analyzing {model.net_g.__class__.__name__}...')

    # analyse Params
    logger.info(f"#Params [M]: {sum(p.numel() for p in model.net_g.parameters() if p.requires_grad)}")

    # analyse FLOPs
    flops = get_model_flops(model.net_g, img_size, False)
    logger.info(f"#FLOPs [G]: {flops / 10 ** 9}")

    # analyse Acts and Conv
    acts, conv = get_model_activation(model.net_g, img_size)
    logger.info(f"#Acts [M]: {acts / 10 ** 6}")
    logger.info(f"#Conv: {conv}")

    # analyse Ave. Time and GPU Mem.
    for analyse_loader in analyse_loaders:
        torch.cuda.reset_peak_memory_stats()
        analyse_set_name = analyse_loader.dataset.opt['name']
        ave_time, gpu_mem = model.nondist_analysis(analyse_loader)
        logger.info(f'#{analyse_set_name} Ave. Time [ms]: {ave_time}')
        logger.info(f'#{analyse_set_name} GPU Mem. [M] in: {gpu_mem}')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    analyse_pipeline(root_path)
