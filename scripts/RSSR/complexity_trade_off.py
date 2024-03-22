import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

method_list = [
    "EDSR",
    "LAPAR",
    "FDIWN",
    "BSRN",  # CNN-based methods
    "SwinIR",
    "ESRT",  # Transformer-based methods
    "CTHN",  # Ours
]


data_dict = [
    ["#93c47d", 177606.0, 28.4173, 0.7111, 117.95, 208.66, 3.0795, 44.081],
    ["#93c47d", 115350.0, 31.0211, 0.8414, 28.525, 162.52, 2.8624, 129.23],
    ["#93c47d", 663848.0, 34.9058, 0.8985, 81.641, 1100.6, 93.112, 16.362],
    ["#93c47d", 352400.0, 35.1295, 0.9045, 31.819, 404.96, 8.9194, 22.150],
    ["#93c47d", 929628.0, 35.1501, 0.904, 218.78, 26.738, 15.71, 37.903],
    ["#93c47d", 751767.0, 34.5005, 0.8898, 75.833, 238.55, 17.004, 32.521],
    ["#ff8800", 687098.0, 35.4333, 0.9071, 26.817, 373.48, 12.426, 14.301],
    ["#c1c1c1", 751767.0, 29.1000, 0.0000, 145.00, 000.00, 17.004, 10.000],
    ["#c1c1c1", 751767.0, 29.1000, 0.0000, 165.00, 000.00, 17.004, 50.000],
    ["#c1c1c1", 751767.0, 29.1000, 0.0000, 185.00, 000.00, 17.004, 100.00],
    ["#c1c1c1", 751767.0, 29.1000, 0.0000, 205.00, 000.00, 17.004, 200.00],
]

color = [data[0] for data in data_dict]
# params = [data[1] for data in data_dict]
psnr = [data[2] for data in data_dict]
# ssim = [data[3] for data in data_dict]
flops = [data[4] for data in data_dict]
# acts = [data[5] for data in data_dict]
# time = [data[6] for data in data_dict]
mem = [data[7] * 20 for data in data_dict]

# 画图
plt.scatter(x=flops, y=psnr, s=mem, c=color, alpha=0.68)

# 标记
plt.text( x=flops[0] -25, y=psnr[0] + 0.2 , s="{}".format(method_list[0]), fontsize=9)
plt.text( x=flops[1] +15, y=psnr[1] - 0.6 , s="{}".format(method_list[1]), fontsize=9)
plt.text( x=flops[2] + 5, y=psnr[2] + 0.2 , s="{}".format(method_list[2]), fontsize=9)
plt.text( x=flops[3] + 7, y=psnr[3] + 0.2 , s="{}".format(method_list[3]), fontsize=9)
plt.text( x=flops[4] -27, y=psnr[4] + 0.2 , s="{}".format(method_list[4]), fontsize=9)
plt.text( x=flops[5] + 5, y=psnr[5] - 0.5 , s="{}".format(method_list[5]), fontsize=9)
plt.text( x=flops[6] + 5, y=psnr[6] + 0.2 , s="{}".format(method_list[6]), fontsize=9)

# 图例
plt.text(x=149, y=0.7867, s="#Memory", fontsize=7)
plt.gca().add_patch(
    Rectangle(
        xy=(148, 0.784),
        width=80,
        height=0.0033,
        fill=False,
        linewidth=0.95,
        linestyle="dashed",
    )
)

plt.text(x=flops[7] - 2.5, y=psnr[7] - 1.0, s="10M", fontsize=7)
plt.text(x=flops[8] - 2.5, y=psnr[8] - 1.0, s="50M", fontsize=7)
plt.text(x=flops[9] - 2.5, y=psnr[9] - 1.0, s="100M", fontsize=7)
plt.text(x=flops[10]- 2.5, y=psnr[10]- 1.0, s="200M", fontsize=7)

# 调整坐标轴
plt.xlabel("FLOPs [G]")
plt.ylabel("PSNR on UCM (agricultural)")
plt.xlim(10, 230)
plt.ylim(28, 36)

# plt.axis([229.6, 34.2, 0.7838, 0.8013])

# MAIN
plt.title("PSNR vs. FLOPs [G] vs. Params [K]")
plt.show()
