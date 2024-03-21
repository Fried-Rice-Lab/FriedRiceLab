# FDIWN
python test.py -expe_opt options/repr/FDIWN/FDIWN_LSR_x2.yml -task_opt options/task/RSSR_x2.yml

# HNCT
python test.py -expe_opt options/repr/HNCT/HNCT_LSR_x2.yml -task_opt options/task/RSSR_x2.yml

# LAPAR
python test.py -expe_opt options/repr/LAPAR/LAPAR-A_LSR_x2.yml -task_opt options/task/RSSR_x2.yml
python test.py -expe_opt options/repr/LAPAR/LAPAR-B_LSR_x2.yml -task_opt options/task/RSSR_x2.yml
python test.py -expe_opt options/repr/LAPAR/LAPAR-C_LSR_x2.yml -task_opt options/task/RSSR_x2.yml

# BSRN
python test.py -expe_opt options/repr/BSRN/BSRN_LSR_x2.yml -task_opt options/task/RSSR_x2.yml

# LBNet
python test.py -expe_opt options/repr/LBNet/LBNet-T_LSR_x2.yml -task_opt options/task/RSSR_x2.yml
