# --------------------------------------------------------------------------------
# Collection of optimization algorithms for PyTorch.
# API and usage patterns are the same as `torch.optim`.
#
# Official GitHub: https://github.com/jettify/pytorch-optimizer
# --------------------------------------------------------------------------------
from .a2grad import A2GradExp, A2GradInc, A2GradUni
from .accsgd import AccSGD
from .adabelief import AdaBelief
from .adabound import AdaBound
from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamod import AdaMod
from .adamp import AdamP
from .aggmo import AggMo
from .apollo import Apollo
from .diffgrad import DiffGrad
from .lamb import Lamb
from .lars import LARS
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .novograd import NovoGrad
from .pid import PID
from .qhadam import QHAdam
from .qhm import QHM
from .radam import RAdam
from .sgdp import SGDP
from .sgdw import SGDW
from .shampoo import Shampoo
from .swats import SWATS
from .yogi import Yogi

__all__ = (
    'A2GradExp',
    'A2GradInc',
    'A2GradUni',
    'AccSGD',
    'AdaBelief',
    'AdaBound',
    'AdaMod',
    'Adafactor',
    'Adahessian',
    'AdamP',
    'AggMo',
    'Apollo',
    'DiffGrad',
    'LARS',
    'Lamb',
    'Lookahead',
    'MADGRAD',
    'NovoGrad',
    'PID',
    'QHAdam',
    'QHM',
    'RAdam',
    'SGDP',
    'SGDW',
    'SWATS',
    'Shampoo',
    'Yogi',
)
