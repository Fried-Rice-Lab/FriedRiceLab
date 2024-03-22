# EDSR
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/EDSR/EDSR_LSR.yml -task_opt options/task/RSSR_x4.yml

# LAPAR
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/LAPAR/LAPAR-C_LSR.yml -task_opt options/task/RSSR_x4.yml

# FDIWN
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/FDIWN/FDIWN_LSR.yml -task_opt options/task/RSSR_x4.yml

# BSRN
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/BSRN/BSRN_LSR.yml -task_opt options/task/RSSR_x4.yml

# HNCT
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/HNCT/HNCT_LSR.yml -task_opt options/task/RSSR_x4.yml

# SwinIR
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/SwinIR/SwinIR_LSR.yml -task_opt options/task/RSSR_x4.yml

# CTHN
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/CTHN/CTHN_LSR.yml -task_opt options/task/RSSR_x4.yml

# ESRT
CUDA_VISIBLE_DEVICES=1 python analyse.py -expe_opt options/expe/ESRT/ESRT_LSR.yml -task_opt options/task/RSSR_x4.yml
