import csv

import matplotlib.pyplot as plt

method_list = ['LatticeNet', 'LAPAR', 'FDIWN',  # CNN-based methods
               'RFDN', 'LBNet', 'ESRT',  # Hybrid-based methods
               'SwinIR', 'ELAN', 'StripWindow']  # Transformer-based methods

with open('/Users/sjp/Documents/研究生文件/SWSRT/数据/1.txt') as f:
    reader = csv.reader(f, dialect=csv.excel_tab)

    data_dict = dict()
    for row in reader:
        if row[0].strip() == 'Model':
            continue
        data_dict[row[0].strip()] = [float(iter.strip()) for iter in row[1:]]

# 准备数据
ssim = [0.7844, 0.7871, 0.7919, 0.7883, 0.7906, 0.7962, 0.7980, 0.7982, 0.8006] + \
       [0.7858, 0.7858, 0.7858, 0.7858, 0.7858] + [0.8010]
params = [data_dict[method][2] for method in method_list] + [data_dict['StripWindow'][2]]
flops = [data_dict[method][3] for method in method_list] + [155, 170, 185, 200, 215] + [data_dict['StripWindow'][3]]
mem = [data_dict[method][5] / 2 for method in method_list] + [100 / 2, 500 / 2, 1000 / 2, 2000 / 2, 3000 / 2] + \
      [data_dict['StripWindow'][5] / 2]
color = ['#93c47d', '#93c47d', '#93c47d',
         '#93c47d', '#4acab4', '#4acab4',
         '#6d9eeb', '#6d9eeb',
         '#ff8800',
         '#c1c1c1', '#c1c1c1', '#c1c1c1', '#c1c1c1', '#c1c1c1'] + ['#ff5500']

# 画图
plt.scatter(x=flops, y=ssim, s=mem, c=color, alpha=0.68)

# 标记
plt.text(x=flops[0] + 5, y=ssim[0], s='{}({}K)'.format(method_list[0], int(params[0] * 1000)), fontsize=9)
plt.text(x=flops[1] + 10, y=ssim[1] + 0.0008, s='{}({}K)'.format(method_list[1], int(params[1] * 1000)), fontsize=9)
plt.text(x=flops[2] + 5, y=ssim[2] + 0.0003, s='{}({}K)'.format(method_list[2], int(params[2] * 1000)), fontsize=9)
plt.text(x=flops[3] + 2, y=ssim[3] + 0.0005, s='{}({}K)'.format(method_list[3], 643), fontsize=9)
plt.text(x=flops[4] - 14, y=ssim[4] + 0.0015, s='{}({}K)'.format(method_list[4], int(params[4] * 1000)), fontsize=9)
plt.text(x=flops[5] + 15, y=ssim[5] - 0.0015, s='{}({}K)'.format(method_list[5], int(params[5] * 1000)), fontsize=9)
plt.text(x=flops[6] - 38, y=ssim[6] + 0.0005, s='{}({}K)'.format(method_list[6], 897), fontsize=9)
plt.text(x=flops[7] + 4, y=ssim[7] + 0.0002, s='{}({}K)'.format(method_list[7], int(params[7] * 1000)), fontsize=9)
plt.text(x=flops[8] + 3, y=ssim[8] - 0.0008, s='{}({}K)'.format('ESWT', int(params[8] * 1000)), fontsize=9)
plt.text(x=flops[8] + 4, y=ssim[-1] - 0.0001, s='{}({}K)'.format('ESWT$^\\dagger$', int(params[8] * 1000)), fontsize=9)

# 图例
plt.text(x=149, y=0.7867, s='#Memory', fontsize=7)
plt.gca().add_patch(plt.Rectangle(xy=(148, 0.784), width=80, height=0.0033,
                                  fill=False, linewidth=0.95, linestyle='dashed'))
plt.text(x=flops[9] - 5.5, y=ssim[9] - 0.0017, s='100M', fontsize=7)
plt.text(x=flops[10] - 5.5, y=ssim[10] - 0.0017, s='500M', fontsize=7)
plt.text(x=flops[11] - 5.5, y=ssim[11] - 0.0017, s='1000M', fontsize=7)
plt.text(x=flops[12] - 5.5, y=ssim[12] - 0.0017, s='2000M', fontsize=7)
plt.text(x=flops[13] - 5.5, y=ssim[13] - 0.0017, s='3000M', fontsize=7)

# 调整坐标轴
plt.xlabel('FLOPs(G)')
plt.ylabel('SSIM on Urban100(x4)')
plt.xlim(34.2, 229.6)
plt.ylim(0.7838, 0.8018)

# plt.axis([229.6, 34.2, 0.7838, 0.8013])

# MAIN
plt.title('SSIM vs. Params[K] vs. FLOPs[G] vs. Memory[M]')
plt.show()
