import copy
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

GR = (1 + np.sqrt(5)) / 2 - 1

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

    # r_min, r_max = np.inf, -np.inf
    # r_grad = [(0, 0), (0, 0)]
    # r_color = 0
    #
    # g_min, g_max = np.inf, -np.inf
    # g_grad = [(0, 0), (0, 0)]
    # g_color = 0
    #
    # b_min, b_max = np.inf, -np.inf
    # b_grad = [(0, 0), (0, 0)]
    # b_color = 0
    #
    # lw = max(im.shape[0], im.shape[1]) / 2
    #
    # for i in range(art.shape[0]):
    #     for j in range(art.shape[1]):
    #         b, g, r = art[i][j]
    #
    #         r_clean = 2 * r - g - b
    #         if r_clean < r_min:
    #             r_min = r_clean
    #             r_grad[0] = j, i
    #         if r_clean > r_max:
    #             r_max = r_clean
    #             r_grad[1] = j, i
    #             r_color = min(1, r / 255), min(1, g / 255), min(1, b / 255), max(.1, min(.5, r / 510))
    #
    #         g_clean = 2 * g - r - b
    #         if g_clean < g_min:
    #             g_min = g_clean
    #             g_grad[0] = j, i
    #         if g_clean > g_max:
    #             g_max = g_clean
    #             g_grad[1] = j, i
    #             g_color = min(1, r / 255), min(1, g / 255), min(1, b / 255), max(.1, min(.5, g / 510))
    #
    #         b_clean = 2 * b - r - g
    #         if b_clean < b_min:
    #             b_min = b_clean
    #             b_grad[0] = j, i
    #         if b_clean > b_max:
    #             b_max = b_clean
    #             b_grad[1] = j, i
    #             b_color = min(1, r / 255), min(1, g / 255), min(1, b / 255), max(.1, min(.5, b / 510))
    #
    # rec = plt.Circle(r_grad[1], max(GR * min(art.shape[0], art.shape[1]),
    #                                 min(abs(r_grad[1][0] - r_grad[0][0]), abs(r_grad[1][1] - r_grad[0][1]))),
    #                  fill=False,
    #                  lw=lw, color=r_color, zorder=2)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(True)
    #
    # rec = plt.Circle(g_grad[1], max(GR * min(art.shape[0], art.shape[1]),
    #                                 min(abs(g_grad[1][0] - g_grad[0][0]), abs(g_grad[1][1] - g_grad[0][1]))),
    #                  fill=False,
    #                  lw=lw, color=g_color, zorder=2)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(True)
    #
    # rec = plt.Circle(b_grad[1], max(GR * min(art.shape[0], art.shape[1]),
    #                                 min(abs(b_grad[1][0] - b_grad[0][0]), abs(b_grad[1][1] - b_grad[0][1]))),
    #                  fill=False,
    #                  lw=lw, color=b_color, zorder=2)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(True)

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

    # r_min, r_max = np.inf, -np.inf
    # r_grad = [(0, 0), (0, 0)]
    # r_color = 0
    #
    # g_min, g_max = np.inf, -np.inf
    # g_grad = [(0, 0), (0, 0)]
    # g_color = 0
    #
    # b_min, b_max = np.inf, -np.inf
    # b_grad = [(0, 0), (0, 0)]
    # b_color = 0
    #
    # lw = 2
    #
    # for i in range(art.shape[0]):
    #     for j in range(art.shape[1]):
    #         b, g, r = art[i][j]
    #
    #         r_clean = 2 * r - g - b
    #         if r_clean < r_min:
    #             r_min = r_clean
    #             r_grad[0] = j, i
    #         if r_clean > r_max:
    #             r_max = r_clean
    #             r_grad[1] = j, i
    #             r_color = min(1, r / 255), min(1, g / 255), min(1, b / 255), max(.1, min(.5, r / 510))
    #
    #         g_clean = 2 * g - r - b
    #         if g_clean < g_min:
    #             g_min = g_clean
    #             g_grad[0] = j, i
    #         if g_clean > g_max:
    #             g_max = g_clean
    #             g_grad[1] = j, i
    #             g_color = min(1, r / 255), min(1, g / 255), min(1, b / 255), max(.1, min(.5, g / 510))
    #
    #         b_clean = 2 * b - r - g
    #         if b_clean < b_min:
    #             b_min = b_clean
    #             b_grad[0] = j, i
    #         if b_clean > b_max:
    #             b_max = b_clean
    #             b_grad[1] = j, i
    #             b_color = min(1, r / 255), min(1, g / 255), min(1, b / 255), max(.1, min(.5, b / 510))
    #
    # rec = plt.Circle(r_grad[1], max(GR * min(art.shape[0], art.shape[1]),
    #                                 min(abs(r_grad[1][0] - r_grad[0][0]), abs(r_grad[1][1] - r_grad[0][1]))),
    #                  fill=False,
    #                  lw=lw, color=r_color, zorder=2)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(True)
    #
    # rec = plt.Circle(g_grad[1], max(GR * min(art.shape[0], art.shape[1]),
    #                                 min(abs(g_grad[1][0] - g_grad[0][0]), abs(g_grad[1][1] - g_grad[0][1]))),
    #                  fill=False,
    #                  lw=lw, color=g_color, zorder=2)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(True)
    #
    # rec = plt.Circle(b_grad[1], max(GR * min(art.shape[0], art.shape[1]),
    #                                 min(abs(b_grad[1][0] - b_grad[0][0]), abs(b_grad[1][1] - b_grad[0][1]))),
    #                  fill=False,
    #                  lw=lw, color=b_color, zorder=2)
    # rec = sub.add_patch(rec)
    # rec.set_clip_on(True)

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


SIZE = 256


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


def train(project: str, n: int, frame: int, threshold: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          dtype: torch.dtype, device: torch.device, losses: [float]) -> int:
    path = IMAGE_DIR / f'{project}.png'
    logging.info(f'Loading {path}')

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im.shape[0] > im.shape[1]:
        im = cv2.resize(im, (round(SIZE / im.shape[0] * im.shape[1]), SIZE))
    else:
        im = cv2.resize(im, (SIZE, round(SIZE / im.shape[1] * im.shape[0])))

    save_image(im, OUTPUT_DIR / project / 'input.png')

    ref_y = torch.tensor(np.array([i for j in im for i in j]), dtype=dtype, device=device)
    ref_shape = im.shape[0], im.shape[1], 3
    logging.info(ref_shape)

    ref_grid = np.mgrid[0:ref_shape[0], 0:ref_shape[1]]
    ref_x = torch.hstack([
        torch.tensor(np.array([[i / (ref_shape[0] - 1)] for i in ref_grid[0].flatten()]), dtype=dtype, device=device),
        torch.tensor(np.array([[i / (ref_shape[1] - 1)] for i in ref_grid[1].flatten()]), dtype=dtype, device=device),
    ])

    logging.info(f'x: {ref_x.shape}')
    logging.info(f'y: {ref_y.shape}')

    # plt.interactive(True)
    fig = plt.figure(figsize=(1920 / DPI, 1080 / DPI), dpi=DPI)
    fig.canvas.draw()

    mse = nn.MSELoss(reduction='mean')
    model.train()

    with tqdm(total=100, position=0, leave=True) as pbar, logging_redirect_tqdm():

        def closure():
            optimizer.zero_grad()
            err = mse(model.forward(ref_x), ref_y)
            err.backward()
            return err

        for i in range(n):
            loss: float | torch.Tensor = optimizer.step(closure)
            losses.append(loss.detach().cpu().numpy())
            progress = int(100 * min(1, max(0, (1 - (np.min(losses) - threshold) / (np.max(losses) - threshold)))))
            change = abs(np.min(losses[-2_000:-1_000]) - np.min(losses[-1_000:])) if i > 2_000 else np.inf
            if pbar.n < progress:
                pbar.update(progress - pbar.n)
            elif i > 2_000 and np.min(losses) < threshold and change < 1:
                logging.info(f'Step {i} (Frame {frame}) -- Loss: {np.min(losses):.12f} ({progress:d}%, {change:.12f})')
                break
            elif i % 10 == 0:
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

                a = art.forward(ref_x).detach().cpu().numpy().reshape(ref_shape)

                save_frame(fig, i, model, model.forward(ref_x).detach().cpu().numpy().reshape(ref_shape), a, losses)
                save_fig(fig, OUTPUT_DIR / project / 'frames' / f'frame_{frame:06d}.png')
                save_art(a, OUTPUT_DIR / project / 'art' / f'frame_{frame:06d}.png')
                frame += 1
            i += 1

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
        art.eval()

        a = art.forward(ref_x).detach().cpu().numpy().reshape(ref_shape)

        save_frame(fig, i, model, model.forward(ref_x).detach().cpu().numpy().reshape(ref_shape), a, losses)
        save_fig(fig, OUTPUT_DIR / project / 'frames' / f'frame_{frame:06d}.png')
        save_art(a, OUTPUT_DIR / project / 'art' / f'frame_{frame:06d}.png')

    model.eval()

    save_image(model.forward(ref_x).detach().cpu().numpy().reshape(ref_shape), OUTPUT_DIR / project / 'output.png')

    eval_shape = 8 * ref_shape[0], 8 * ref_shape[1], ref_shape[2]
    eval_grid = np.mgrid[0:eval_shape[0], 0:eval_shape[1]]

    eval_x = torch.hstack([
        torch.tensor(np.array([[i / (eval_shape[0] - 1)] for i in eval_grid[0].flatten()]), dtype=dtype, device=device),
        torch.tensor(np.array([[i / (eval_shape[1] - 1)] for i in eval_grid[1].flatten()]), dtype=dtype, device=device),
    ])

    save_image(model.forward(eval_x).detach().cpu().numpy().reshape(eval_shape), OUTPUT_DIR / project / 'eval.png')
    save_art(art.forward(eval_x).detach().cpu().numpy().reshape(eval_shape), OUTPUT_DIR / project / 'art.png')

    return frame


def main() -> None:
    assert torch.cuda.is_available()

    device = torch.device('cuda')
    dtype = torch.float32
    layer = SIZE

    for project in PROJECTS:
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

        n = 1_000_000

        optimizer = torch.optim.Adam(model.parameters())
        threshold = 100
        frame = 1
        losses = []

        train(project, n, frame, threshold, model, optimizer, dtype, device, losses)


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
