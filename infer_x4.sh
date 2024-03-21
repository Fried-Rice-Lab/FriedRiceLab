# FDIWN
python infer.py -expe_opt options/repr/FDIWN/FDIWN_LSR.yml -task_opt options/task/RSSR_x4.yml

# HNCT
python infer.py -expe_opt options/repr/HNCT/HNCT_LSR.yml -task_opt options/task/RSSR_x4.yml

# LAPAR
python infer.py -expe_opt options/repr/LAPAR/LAPAR-A_LSR.yml -task_opt options/task/RSSR_x4.yml
python infer.py -expe_opt options/repr/LAPAR/LAPAR-B_LSR.yml -task_opt options/task/RSSR_x4.yml
python infer.py -expe_opt options/repr/LAPAR/LAPAR-C_LSR.yml -task_opt options/task/RSSR_x4.yml

# BSRN
python infer.py -expe_opt options/repr/BSRN/BSRN_LSR.yml -task_opt options/task/RSSR_x4.yml
