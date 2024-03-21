# FDIWN
python test.py -expe_opt options/repr/FDIWN/FDIWN_LSR.yml -task_opt options/task/RSSR_x4.yml

# HNCT
python test.py -expe_opt options/repr/HNCT/HNCT_LSR.yml -task_opt options/task/RSSR_x4.yml

# LAPAR
python test.py -expe_opt options/repr/LAPAR/LAPAR-A_LSR.yml -task_opt options/task/RSSR_x4.yml
python test.py -expe_opt options/repr/LAPAR/LAPAR-B_LSR.yml -task_opt options/task/RSSR_x4.yml
python test.py -expe_opt options/repr/LAPAR/LAPAR-C_LSR.yml -task_opt options/task/RSSR_x4.yml

# BSRN
python test.py -expe_opt options/repr/BSRN/BSRN_LSR.yml -task_opt options/task/RSSR_x4.yml

# LBNet
python test.py -expe_opt options/repr/LBNet/LBNet-T_LSR.yml -task_opt options/task/RSSR_x4.yml
