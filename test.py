from os import path as osp

from basicsr.test import test_pipeline

import archs  # noqa
import data  # noqa
import models  # noqa

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
