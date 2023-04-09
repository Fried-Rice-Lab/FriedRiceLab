# ----------------------------------------------------------------------
# re parameterize for EFDN
# Modified from https://github.com/DingXiaoH/RepVGG
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ----------------------------------------------------------------------
import argparse
import copy
import os
import sys

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

sys.path.append('./')
from archs.efdn_arch import EFDN

parser = argparse.ArgumentParser(description='EFDN Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='EFDN')


def model_convert(model: torch.nn.Module, save_path=None):
    model = copy.deepcopy(model)

    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    save_dict = {}
    model = model if isinstance(model, list) else [model]

    for model_ in model:
        state_dict = model_.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            state_dict[key] = param.cpu()
        save_dict['params'] = state_dict

    torch.save(save_dict, save_path)
    return save_dict


def convert():
    args = parser.parse_args()

    train_model = EFDN(upscale=4, n_feats=48, num_in_ch=3, num_out_ch=3, task='lsr', deploy=False)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)['params']

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']

        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()
