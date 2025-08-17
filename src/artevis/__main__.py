import argparse
import copy
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src import OUTPUT_DIR, SEISMIC, SEISMIC_POSITIVE, SEISMIC_NEGATIVE, IMAGE_DIR
from src.artevis import DPI, DEFAULT_SIZE, PROJECTS, DEFAULT_N, THRESHOLD, DTYPE, DEFAULT_FPS


def save_fig(fig: plt.Figure, path: Path, dpi: float = DPI) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], transparent=False, dpi=dpi)


def save_image(im: np.ndarray, path: Path) -> None:
    im = im.astype(np.int32)
    im[im > 255] = 255
    im[im < 0] = 0

    fig = plt.Figure(figsize=(im.shape[1], im.shape[0]), dpi=1)

    sub = fig.add_subplot()
    sub.set_axis_off()
    sub.imshow(im[:, :, ::-1], interpolation='nearest')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    sub.margins(0, 0)
    save_fig(fig, path, 1)


def save_art(art: np.ndarray, path: Path) -> None:
    w, b = np.nanmin(art), np.nanmax(art)
    art -= w
    art *= 255 / (b - w)
    im = art.astype(np.int32)

    im[im > 255] = 255
    im[im < 0] = 0

    fig = plt.Figure(figsize=(im.shape[1], im.shape[0]), dpi=1)

    sub = fig.add_subplot()
    sub.set_axis_off()
    sub.imshow(im[:, :, ::-1], interpolation='nearest', zorder=1)

    sub.set_xlim(0, im.shape[1])
    sub.set_ylim(0, im.shape[0])

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    sub.margins(0, 0)
    save_fig(fig, path, 1)


def save_frame(fig: plt.Figure, n: int, model: torch.nn.Module, im: np.ndarray, art: np.ndarray,
               losses: [float]) -> None:
    weights = [w.detach().cpu().numpy() for i, w in model.named_parameters() if 'weight' in i][1:-1]

    im = im.astype(np.int32)
    im[im > 255] = 255
    im[im < 0] = 0

    w, b = np.nanmin(art), np.nanmax(art)
    art -= w
    art *= 255 / (b - w)
    art = art.astype(np.int32)

    fig.clear()

    sub = fig.add_subplot(2, 1, 2)
    sub.set_axis_off()
    sub.set_yscale('log', base=10)
    sub.plot(losses[-5_000:], c='r')

    sub = fig.add_subplot(2, len(weights) + 2, len(weights) + 1)
    sub.set_axis_off()
    sub.imshow(im[:, :, ::-1], interpolation='nearest', zorder=1)

    sub = fig.add_subplot(2, len(weights) + 2, len(weights) + 2)
    sub.set_axis_off()
    sub.imshow(art[:, :, ::-1], interpolation='nearest', zorder=1)
    sub.set_xlim(0, art.shape[1])
    sub.set_ylim(0, art.shape[0])

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
        sub = fig.add_subplot(2, len(weights) + 2, i + 1)
        sub.set_axis_off()
        sub.set_frame_on(True)
        sub.imshow(w, cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

    fig.suptitle(f'{n}\n({np.min(losses):.3f})', fontsize=14)
    fig.subplots_adjust(bottom=.1, top=.9, left=0.02, right=.98, wspace=.05, hspace=.05)

    fig.canvas.flush_events()


def artsy(weights: [torch.Tensor], biases: [torch.Tensor]):
    weights = [
        weights[0],
    ] + [
        weights[3].rot90(),
        weights[2].rot90().rot90(),
        weights[1].rot90().rot90().rot90(),
    ] + [
        weights[4],
    ]
    biases = [
        biases[0],
    ] + [
        biases[3],
        biases[2],
        biases[1],
    ] + [
        biases[4],
    ]
    return weights, biases


def image(project: str, size: int) -> np.ndarray:
    path = IMAGE_DIR / f'{project}.png'
    logging.info(f'Loading Image {path}')

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im.shape[0] > im.shape[1]:
        im = cv2.resize(im, (round(size / im.shape[0] * im.shape[1]), size))
    else:
        im = cv2.resize(im, (size, round(size / im.shape[1] * im.shape[0])))

    return im


def train(project: str, im: np.ndarray, n: int, threshold: float, fps: float, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, device: torch.device, losses: [float]) -> None:
    save_image(im, OUTPUT_DIR / project / 'input.png')

    y = torch.tensor(np.array([i for j in im for i in j]), dtype=DTYPE, device=device)
    shape = im.shape[0], im.shape[1], 3
    logging.info(shape)

    grid = np.mgrid[0:shape[0], 0:shape[1]]
    x = torch.hstack([
        torch.tensor(np.array([[i / (shape[0] - 1)] for i in grid[0].flatten()]), dtype=DTYPE, device=device),
        torch.tensor(np.array([[i / (shape[1] - 1)] for i in grid[1].flatten()]), dtype=DTYPE, device=device),
    ])

    logging.info(f'x: {x.shape}')
    logging.info(f'y: {y.shape}')

    # plt.interactive(True)
    fig = plt.figure(figsize=(1920 / DPI, 1080 / DPI), dpi=DPI)
    fig.canvas.draw()

    mse = nn.MSELoss(reduction='mean')
    model.train()

    logging.info(f'Training -- Start')

    frame = 1
    with tqdm(total=100, position=0, leave=True) as pbar, logging_redirect_tqdm():

        def closure():
            optimizer.zero_grad()
            err = mse(model.forward(x), y)
            err.backward()
            return err

        for i in range(n):
            loss: float | torch.Tensor = optimizer.step(closure)
            losses.append(loss.detach().cpu().numpy())
            progress = int(100 * min(1, max(0, (1 - (np.min(losses) - threshold) / (np.max(losses) - threshold)))))
            change = abs(np.min(losses[-2_000:-1_000]) - np.min(losses[-1_000:])) if i > 2_000 else np.inf

            if pbar.n < progress:
                pbar.update(progress - pbar.n)

            fin = i > 2_000 and np.min(losses) < threshold and change < 1

            if i % (600 // fps) == 0 or fin:
                logging.info(f'Step {i} (Frame {frame}) -- Loss: {np.min(losses):.12f} ({progress:d}%, {change:.12f})')

                art = copy.deepcopy(model)
                weights, biases = [], []
                for name, param in art.named_parameters():
                    if 'weight' in name:
                        weights.append(param.detach())
                    if 'bias' in name:
                        biases.append(param.detach())

                weights, biases = artsy(weights, biases)

                for name, param in art.named_parameters():
                    if 'weight' in name:
                        param.data = nn.parameter.Parameter(weights.pop(0))
                    if 'bias' in name:
                        param.data = nn.parameter.Parameter(biases.pop(0))

                art.to(device)
                art.eval()

                a = art.forward(x).detach().cpu().numpy().reshape(shape)
                save_frame(fig, i, model, model.forward(x).detach().cpu().numpy().reshape(shape), a, losses)
                save_fig(fig, OUTPUT_DIR / project / 'frames' / f'frame_{frame:06d}.png')
                save_art(a, OUTPUT_DIR / project / 'art' / f'frame_{frame:06d}.png')
                frame += 1

            if fin:
                break

    logging.info(f'Training -- End')

    save_image(model.forward(x).detach().cpu().numpy().reshape(shape), OUTPUT_DIR / project / 'output.png')


def render(project: str, im: np.ndarray, model: torch.nn.Module, device: torch.device) -> None:
    logging.info(f'Rendering -- Start')

    art = copy.deepcopy(model)
    weights, biases = [], []
    for name, param in art.named_parameters():
        if 'weight' in name:
            weights.append(param.detach())
        elif 'bias' in name:
            biases.append(param.detach())

    weights, biases = artsy(weights, biases)

    for name, param in art.named_parameters():
        if 'weight' in name:
            param.data = nn.parameter.Parameter(weights.pop(0))
        elif 'bias' in name:
            param.data = nn.parameter.Parameter(biases.pop(0))

    art.to(device)

    shape = 8 * im.shape[0], 8 * im.shape[1], 3
    grid = np.mgrid[0:shape[0], 0:shape[1]]

    x = torch.hstack([
        torch.tensor(np.array([[i / (shape[0] - 1)] for i in grid[0].flatten()]), dtype=DTYPE, device=device),
        torch.tensor(np.array([[i / (shape[1] - 1)] for i in grid[1].flatten()]), dtype=DTYPE, device=device),
    ])

    save_image(model.forward(x).detach().cpu().numpy().reshape(shape), OUTPUT_DIR / project / 'eval.png')
    save_art(art.forward(x).detach().cpu().numpy().reshape(shape), OUTPUT_DIR / project / 'art.png')

    logging.info(f'Rendering -- End')


def main(options: argparse.Namespace) -> None:
    device = torch.device(options.device)
    layer = options.size

    model = nn.Sequential()
    model.append(nn.Linear(2, layer, bias=True, dtype=DTYPE))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=DTYPE))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=DTYPE))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=DTYPE))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, 3, bias=True, dtype=DTYPE))
    logging.info(model)

    n = options.steps
    path = OUTPUT_DIR / options.project / 'tensor.pt'
    path.parent.mkdir(parents=True, exist_ok=True)

    if options.cache and path.exists():
        logging.info(f'Loading Model {path}')
        model.load_state_dict(torch.load(path, weights_only=True))
        n = 0

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    threshold = THRESHOLD

    im = image(options.project, options.size)

    if n > 0:
        model.train()
        train(options.project, im, n, threshold, options.fps, model, optimizer, device, [])

    model.eval()
    render(options.project, im, model, device)

    if options.cache:
        logging.info(f'Saving Model {path}')
        torch.save(model.state_dict(), path)


def cmd() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='artevis')

    parser.add_argument(
        '--cache',
        action='store_true',
        default=False,
        help='Enable caching / storage of models in output folder',
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='device used for training (default: cpu)',
    )
    parser.add_argument(
        '--steps',
        type=int,
        metavar='<steps>',
        default=DEFAULT_N,
        help=f'number of training steps (default: {DEFAULT_N:.0e})',
    )
    parser.add_argument(
        '--fps',
        type=float,
        metavar='fps',
        default=DEFAULT_FPS,
        help=f'FPS (default: {DEFAULT_FPS:.0e})',
    )
    parser.add_argument(
        '--size',
        type=int,
        metavar='<size>',
        default=DEFAULT_SIZE,
        help=f'size of trained image (default: {DEFAULT_SIZE})',
    )
    parser.add_argument(
        '--project',
        type=str,
        metavar='<project>',
        choices=PROJECTS,
        default=PROJECTS[0],
        help=f'choose project (default: {PROJECTS[0]})',
    )

    options = parser.parse_args()

    assert options.steps > 0
    assert options.size > 0
    assert options.fps > 0
    if options.device == 'cuda':
        assert torch.cuda.is_available()

    return options


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(module)s[%(levelname)s]: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)],
                        encoding='utf-8',
                        level=logging.INFO)
    try:
        main(cmd())
        logging.info('EXIT -- Success')
    except KeyboardInterrupt:
        logging.info('EXIT -- Abort')
