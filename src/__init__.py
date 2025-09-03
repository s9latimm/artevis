import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import colors, pyplot as plt

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = .5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = .5

GRAY: plt.Colormap = colors.LinearSegmentedColormap.from_list('gray', plt.get_cmap('gray')(np.linspace(0, 1., 100)))
SEISMIC: plt.Colormap = colors.LinearSegmentedColormap.from_list('seismic',
                                                                 plt.get_cmap('seismic')(np.linspace(0, 1., 100)))
SEISMIC_NEGATIVE: plt.Colormap = colors.LinearSegmentedColormap.from_list(
    'seismic_neg',
    plt.get_cmap('seismic')(np.linspace(0., .5, 50)))
SEISMIC_POSITIVE: plt.Colormap = colors.LinearSegmentedColormap.from_list(
    'seismic_pos',
    plt.get_cmap('seismic')(np.linspace(.5, 1., 50)))

GOLDEN_RATIO = 1 / (1 + np.sqrt(5)) / 2

ROOT_DIR: Path = Path(__file__).parent.parent.absolute()
OUTPUT_DIR: Path = ROOT_DIR / 'output'
IMAGE_DIR: Path = ROOT_DIR / 'images'
