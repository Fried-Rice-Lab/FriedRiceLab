import os.path as osp

from basicsr.train import train_pipeline

import archs  # noqa
import data  # noqa
import models  # noqa

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
