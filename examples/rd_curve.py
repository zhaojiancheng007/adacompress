import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
from matplotlib.ticker import AutoMinorLocator

params = {
    "font.size": 12,
    'font.family':'Liberation Sans',
    "figure.subplot.wspace": 0.2,
    "figure.subplot.hspace": 0.4,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "axes.linewidth": 1.2  # 加粗边框
}
plt.rcParams.update(params)

all_tasks = [
    {
        'name': 'Semantic Segmentation',
        'ylabel': 'mIoU (%) ↑',
        'baseline': 68.38,
        'data': {
            'Ours': [(0.1107, 61.15), (0.1751, 64.16),(0.2452, 65.40),(0.3401, 66.45)],
            'Full Finetune': [(0.0879, 58.64), (0.1266, 60.69), (0.2171, 63.00), (0.3209, 64.26)],
            'AdaptICMH(Adaptor)': [(0.0795, 57.65), (0.1086, 60.71), (0.2088, 64.83), (0.3257, 65.06)],
            'TransTIC(VPT)': [(0.0795, 57.25), (0.1086, 60.01), (0.2088, 64.03), (0.3257, 64.88)],
        }
    },
    {
        'name': 'Human Parsing',
        'ylabel': 'mIoU (%) ↑',
        'baseline': 57.22,
        'data': {
            'Ours': [(0.1106, 53.70), (0.1614, 54.92),(0.2309, 55.84),(0.3338, 56.43)],
            'Full Finetune': [(0.0877, 52.04), (0.1483, 54.39), (0.2057, 55.18), (0.2661 , 55.95),(0.3205, 56.29)],
            'AdaptICMH(Adaptor)': [(0.0942, 51.65), ( 0.1905, 54.85), (0.3259, 56.18)],
            'TransTIC(VPT)': [(0.0795, 45.7), (0.1086, 49.6), (0.2088, 54.2), (0.3257, 54.8)],
        }
    },
    {
        'name': 'Saliency Detection',
        'ylabel': 'mIoU (%) ↑',
        'baseline': 63.04,
        'data': {
            'Ours': [(0.1106, 61.80), (0.1614, 62.32),(0.2309, 62.82),(0.3338, 62.97)],
            'Full Finetune': [(0.0846, 59.52), (0.1153, 59.90), (0.1752, 60.48), (0.2681, 60.86)],
            'AdaptICMH(Adaptor)': [(0.1229, 60.31), (0.1457, 60.79), (0.1825, 61.25), (0.2677, 61.15), (0.3994, 61.25)],
            'TransTIC(VPT)': [(0.0879, 59.14), (0.1229, 60.31), (0.1439, 60.83), (0.3087, 60.85)],
        }
    },
    {
        'name': 'Normals Estimation',
        'ylabel': 'RMSE ↓',
        'baseline': 17.21,
        'data': {
            'Ours': [(0.1107, 16.81), (0.1751, 16.63),(0.2452, 16.58),(0.3401, 16.51)],
            'Full Finetune': [(0.1015, 16.96), (0.1505, 16.81),(0.2184, 16.70),(0.3644, 16.66)],
            'AdaptICMH(Adaptor)': [(0.1140, 16.75), (0.1691, 16.63), (0.2365, 16.55), (0.3248, 16.51)],
            # 'AdaptICMH(Adaptor)': [(0.1221, 17.28), (0.1691 , 16.93), (0.2658, 16.89), (0.3819, 16.86)],
            'TransTIC(VPT)': [(0.1114, 16.80), (0.1468, 16.65), (0.1761, 16.60)],
        }
    }
]

# 样式设置
style_dict = {
    'Ours': dict(linestyle='-', marker='*', markersize=8, linewidth=2, color='#d7191c'),
    'Full Finetune': dict(linestyle=':', marker='o', markersize=6, linewidth=2, color='#fdae61'),
    'AdaptICMH(Adaptor)': dict(linestyle='--', marker='^', markersize=6, linewidth=2, color='#2c7bb6'),
    'TransTIC(VPT)': dict(linestyle='-.', marker='s', markersize=6, linewidth=2, color='#91bfdb'),
}

fig, ax = plt.subplots(figsize=(7, 5))

for task in all_tasks:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = task['data']
    baseline_y = task['baseline']
    y_label = task['ylabel']

    x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')

    for label, points in data.items():
        x_data, y_data = zip(*points)
        x_min, x_max = min(x_min, min(x_data)), max(x_max, max(x_data))
        y_min, y_max = min(y_min, min(y_data)), max(y_max, max(y_data))
        ax.plot(x_data, y_data, label=label, **style_dict[label])

    y_min, y_max = min(y_min, baseline_y), max(y_max, baseline_y)
    ax.set_xlim(x_min - 0.02, x_max + 0.02)
    ax.set_ylim(y_min - 0.05 * abs(y_max - y_min), y_max + 0.05 * abs(y_max - y_min))

    ax.axhline(y=baseline_y, color='gray', linestyle='--', linewidth=2)
    ax.text(x_max, baseline_y + 0.01 * abs(y_max - y_min), 'Uncompressed',
            color='gray', ha='right', va='bottom', fontsize=12)

    ax.set_xlabel('Bit-rate (bpp)')
    ax.set_ylabel(y_label)
    ax.set_title(task['name'], pad=10)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(loc='best', frameon=False, ncol=1)

    # 保存 PNG
    filename = task['name'].replace(" ", "_").lower()
    fig.savefig(f"./rdcurve/{filename}_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig)