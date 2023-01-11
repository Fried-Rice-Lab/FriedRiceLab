import logging
import os.path
from os import path as osp

from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str

import archs  # noqa
import data  # noqa
import models  # noqa
from tools.interpret_tool import get_model_interpretation
from utils import parse_options


def interpret_pipeline(root_path):  # noqa
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"interpret_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = build_model(opt)

    logger.info(f'Interpreting {model.net_g.__class__.__name__}...')

    for _, img_opt in sorted(opt['interpret_imgs'].items()):
        img, di = get_model_interpretation(model.net_g, img_opt['img_path'], img_opt['w'], img_opt['h'],
                                           use_cuda=True if opt['num_gpu'] > 0 else False)
        img.save(osp.join(opt['path']['visualization'], os.path.basename(img_opt['img_path'])))
        logger.info(f"DI of {os.path.basename(img_opt['img_path'])}: {round(di, 3)}")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    interpret_pipeline(root_path)
