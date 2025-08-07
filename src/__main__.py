import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from matplotlib import colors
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'

DPI: int = 1000
SCALE: float = 2

GRAY: plt.Colormap = colors.LinearSegmentedColormap.from_list('gray',
                                                              plt.get_cmap('gray')(np.linspace(0, 1., 100)))
SEISMIC: plt.Colormap = colors.LinearSegmentedColormap.from_list('seismic',
                                                                 plt.get_cmap('seismic')(np.linspace(0, 1., 100)))
SEISMIC_NEGATIVE: plt.Colormap = colors.LinearSegmentedColormap.from_list(
    'seismic_neg',
    plt.get_cmap('seismic')(np.linspace(0., .5, 50)))
SEISMIC_POSITIVE: plt.Colormap = colors.LinearSegmentedColormap.from_list(
    'seismic_pos',
    plt.get_cmap('seismic')(np.linspace(.5, 1., 50)))

OUTPUT_DIR: Path = Path(__file__).parents[1] / 'output'
IMAGE_DIR: Path = Path(__file__).parents[1] / 'images'
PROJECT = 'mona-lisa'
PROJECT_DIR = OUTPUT_DIR / PROJECT


def save_fig(fig: plt.Figure, path: Path) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], transparent=False, dpi=DPI)


def save_image(im: np.ndarray, filename: str) -> None:
    im[im > 255] = 255
    im[im < 0] = 0

    norm = colors.Normalize(vmin=0, vmax=255)
    cmap = GRAY

    fig = plt.Figure(figsize=(1080 / DPI, 1080 / DPI))
    sub = fig.add_subplot()
    sub.set_axis_off()
    sub.imshow(im, cmap=cmap, norm=norm, interpolation='nearest')
    save_fig(fig, PROJECT_DIR / f'{filename}.png')


def save_frame(n: int, suffix: int, model: torch.nn.Module, im: np.ndarray, loss: float) -> None:
    weights = [w.detach().cpu().numpy() for i, w in model.named_parameters() if 'weight' in i][1:-1]
    fig = plt.Figure((1920 / DPI, 1080 / DPI))

    im[im > 255] = 255
    im[im < 0] = 0

    norm = colors.Normalize(vmin=0, vmax=255)
    cmap = GRAY

    sub = fig.add_subplot(1, len(weights) + 1, len(weights) + 1)
    sub.set_axis_off()
    sub.imshow(im, cmap=cmap, norm=norm, interpolation='nearest',  zorder=1)
    ax = sub.axis()
    rec = plt.Rectangle((ax[0], ax[2]), ax[1] - ax[0], ax[3] - ax[2], fill=False, lw=.8,
                        linestyle='solid', zorder=0)
    rec = sub.add_patch(rec)
    rec.set_clip_on(False)

    vmin, vmax = np.nanmin(weights), np.nanmax(weights)
    if vmin < 0 < vmax:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = SEISMIC
    elif vmax == vmin:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = SEISMIC
    elif vmax < 0:
        norm = colors.Normalize(vmin=vmin, vmax=0)
        cmap = SEISMIC_NEGATIVE
    else:
        norm = colors.Normalize(vmin=0, vmax=vmax)
        cmap = SEISMIC_POSITIVE

    for i, w in enumerate(weights):
        sub = fig.add_subplot(1, len(weights) + 1, i + 1)
        sub.set_axis_off()
        sub.set_frame_on(True)
        sub.imshow(w, cmap=cmap, norm=norm, interpolation='nearest', zorder=1)
        ax = sub.axis()
        rec = plt.Rectangle((ax[0], ax[2]), ax[1] - ax[0], ax[3] - ax[2], fill=False, lw=.8,
                            linestyle='solid', zorder=0)
        rec = sub.add_patch(rec)
        rec.set_clip_on(False)

    fig.suptitle(f'{n}\n({loss:.3f})', fontsize=2)
    fig.subplots_adjust(bottom=.05, top=.95, left=0.05, right=.95, wspace=.05, hspace=.05)
    save_fig(fig, PROJECT_DIR / f'frame_{suffix:06d}.png')


def main() -> None:
    assert torch.cuda.is_available()

    device = torch.device('cuda')

    image = IMAGE_DIR / f'{PROJECT}.png'
    logging.debug(f'Loading {image}')
    im = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)

    save_image(im, 'input')

    dtype = torch.float32

    y = torch.tensor([[i] for i in im.flatten()], dtype=dtype, device=device)

    shape = im.shape
    grid = np.mgrid[0:shape[0], 0:shape[1]]

    x = torch.hstack([
        torch.tensor([[i] for i in grid[0].flatten()], dtype=dtype, device=device),
        torch.tensor([[i] for i in grid[1].flatten()], dtype=dtype, device=device),
    ])

    logging.debug(f'x: {x.shape}')
    logging.debug(f'y: {y.shape}')

    layer = 100

    model = nn.Sequential()
    model.append(nn.Linear(2, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, 1, bias=True, dtype=dtype))
    logging.info(model)
    model.to(device)

    n = 500_000
    optimizer = torch.optim.Adam(model.parameters())

    mse = nn.MSELoss()
    delta = 1e-2
    frame = 1

    model.train()
    with tqdm(total=n, position=0, leave=True) as pbar, logging_redirect_tqdm():

        def closure():
            optimizer.zero_grad()
            err = mse(model.forward(x), y)
            err.backward()
            return err

        for _ in range(n):
            loss = optimizer.step(closure)
            if loss < delta:
                if pbar.n < n:
                    pbar.update(1)
                break
            if pbar.n % 100 == 0:
                logging.debug(f'Loss: {loss:.12f}')
                save_frame(pbar.n, frame, model, model.forward(x).detach().cpu().numpy().reshape(shape), loss)
                frame += 1
            if pbar.n < n:
                pbar.update(1)

        logging.debug(f'Loss: {loss:.12f}')
        save_frame(pbar.n, frame, model, model.forward(x).detach().cpu().numpy().reshape(shape), loss)

    model.eval()
    with torch.no_grad():
        save_image(model.forward(x).detach().cpu().numpy().reshape(shape), 'output')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(module)s[%(levelname)s]: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)],
                        encoding='utf-8',
                        level=logging.DEBUG)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    try:
        main()
        logging.info('EXIT -- Success')
    except KeyboardInterrupt:
        logging.info('EXIT -- Abort')
