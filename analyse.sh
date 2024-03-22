# EDSR
python analyse.py -expe_opt options/repr/EDSR/EDSR_LSR.yml -task_opt options/task/RSSR_x4.yml

# LAPAR
python analyse.py -expe_opt options/repr/LAPAR/LAPAR-C_LSR.yml -task_opt options/task/RSSR_x4.yml

# FDIWN
python analyse.py -expe_opt options/repr/FDIWN/FDIWN_LSR.yml -task_opt options/task/RSSR_x4.yml

# BSRN
python analyse.py -expe_opt options/repr/BSRN/BSRN_LSR.yml -task_opt options/task/RSSR_x4.yml

# HNCT
python analyse.py -expe_opt options/repr/HNCT/HNCT_LSR.yml -task_opt options/task/RSSR_x4.yml

# SwinIR
python analyse.py -expe_opt options/repr/SwinIR/SwinIR_LSR.yml -task_opt options/task/RSSR_x4.yml
