import logging
import random
import sys
from pathlib import Path

import matplotlib
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

matplotlib.use('TkAgg')

DPI: int = 100
SCALE: float = 2

GRAY: plt.Colormap = colors.LinearSegmentedColormap.from_list('gray', plt.get_cmap('gray')(np.linspace(0, 1., 100)))
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
PROJECTS = ['mona-lisa_1080', 'girl_1080', 'nebelmeer_1080', 'schrei_1080', 'sterne_1080']


def save_fig(fig: plt.Figure, path: Path) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], transparent=False, dpi=DPI)


def save_image(im: np.ndarray, filename: str) -> None:
    im = im.astype(np.int32)
    im[im > 255] = 255
    im[im < 0] = 0

    fig = plt.Figure(figsize=(1080 / DPI, 1080 / DPI))
    sub = fig.add_subplot()
    sub.set_axis_off()
    sub.imshow(im[:, :, ::-1], interpolation='nearest')
    save_fig(fig, OUTPUT_DIR / f'{filename}.png')


def save_frame(fig: plt.Figure, n: int, model: torch.nn.Module, im: np.ndarray, losses: [float]) -> None:
    weights = [w.detach().cpu().numpy() for i, w in model.named_parameters() if 'weight' in i][1:-1]

    im = im.astype(np.int32)
    im[im > 255] = 255
    im[im < 0] = 0

    fig.clear()

    sub = fig.add_subplot(2, 1, 2)
    sub.set_axis_off()
    sub.plot(losses[-1000:], c='r')

    sub = fig.add_subplot(2, len(weights) + 1, len(weights) + 1)
    sub.set_axis_off()
    sub.imshow(im[:, :, ::-1], interpolation='nearest', zorder=1)
    # ax = sub.axis()
    # rec = plt.Rectangle((ax[0], ax[2]), ax[1] - ax[0], ax[3] - ax[2], fill=False, lw=.8,
    #                     linestyle='solid', zorder=0)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(False)

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
        sub = fig.add_subplot(2, len(weights) + 1, i + 1)
        sub.set_axis_off()
        sub.set_frame_on(True)
        sub.imshow(w, cmap=cmap, norm=norm, interpolation='nearest', zorder=1)
        # ax = sub.axis()
        # rec = plt.Rectangle((ax[0], ax[2]), ax[1] - ax[0], ax[3] - ax[2], fill=False, lw=.8,
        #                     linestyle='solid', zorder=0)
        # rec = sub.add_patch(rec)
        # rec.set_clip_on(False)

    fig.suptitle(f'{n}\n({np.min(losses):.3f})', fontsize=14)
    fig.subplots_adjust(bottom=.1, top=.9, left=0.02, right=.98, wspace=.05, hspace=.05)

    fig.canvas.flush_events()


SIZE = 256


def train(image: Path, n: int, frame: int, delta: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          dtype, device, losses) -> int:
    logging.info(f'Loading {image}')

    im = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if im.shape[0] > im.shape[1]:
        im = cv2.resize(im, (round(SIZE / im.shape[0] * im.shape[1]), SIZE))
    else:
        im = cv2.resize(im, (SIZE, round(SIZE / im.shape[1] * im.shape[0])))

    logging.info(im.shape)
    save_image(im, 'input')

    y = torch.tensor([i for j in im for i in j], dtype=dtype, device=device)
    shape = (im.shape[0], im.shape[1], 3)

    grid = np.mgrid[0:shape[0], 0:shape[1]]
    mse = nn.MSELoss(reduction='mean')

    x = torch.hstack([
        torch.tensor([[i + 1] for i in grid[0].flatten()], dtype=dtype, device=device),
        torch.tensor([[i + 1] for i in grid[1].flatten()], dtype=dtype, device=device),
    ])

    logging.info(f'x: {x.shape}')
    logging.info(f'y: {y.shape}')

    # plt.interactive(True)
    fig = plt.figure(figsize=(1920 / DPI, 1080 / DPI), dpi=DPI)
    fig.canvas.draw()
    step = 0

    model.train()

    with tqdm(total=100, position=0, leave=True) as pbar, logging_redirect_tqdm():

        def closure():
            optimizer.zero_grad()
            err = mse(model.forward(x), y)
            err.backward()
            return err

        for _ in range(n):
            loss = optimizer.step(closure)
            losses.append(loss.detach().cpu().numpy())
            progress = int(100 * min(1, max(0, (1 - (np.min(losses) - delta) / (np.max(losses) - delta)))))
            pbar.update(progress - pbar.n)
            if loss < delta:
                logging.info(f'Step {step} (Frame {frame}) -- Loss: {np.min(losses):.12f} ({progress:d}%)')
                break
            if step % 20 == 0:
                logging.info(f'Step {step} (Frame {frame}) -- Loss: {np.min(losses):.12f} ({progress:d}%)')
                save_frame(fig, pbar.n, model, model.forward(x).detach().cpu().numpy().reshape(shape), losses)
                save_fig(fig, OUTPUT_DIR / f'frame_{frame:06d}.png')
                frame += 1
            step += 1

        save_frame(fig, pbar.n, model, model.forward(x).detach().cpu().numpy().reshape(shape), losses)
        save_fig(fig, OUTPUT_DIR / f'frame_{frame:06d}.png')

    model.eval()

    save_image(model.forward(x).detach().cpu().numpy().reshape(shape), 'output.png')
    return frame


def main() -> None:
    assert torch.cuda.is_available()

    device = torch.device('cuda')
    dtype = torch.float32
    layer = SIZE

    model = nn.Sequential()
    model.append(nn.Linear(2, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=dtype))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, 3, bias=True, dtype=dtype))
    logging.info(model)

    model.to(device)

    n = np.iinfo(np.int32).max
    # n = 1_000_000
    optimizer = torch.optim.Adam(model.parameters())
    delta = 100
    frame = 1
    losses = []

    project = PROJECTS[1]
    train(IMAGE_DIR / f'{project}.png', n, frame, delta, model, optimizer, dtype, device, losses)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(module)s[%(levelname)s]: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)],
                        encoding='utf-8',
                        level=logging.INFO)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    try:
        main()
        logging.info('EXIT -- Success')
    except KeyboardInterrupt:
        logging.info('EXIT -- Abort')
