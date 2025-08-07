import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DPI: int = 1000
SCALE: float = 2.5


def save_fig(fig: plt.Figure, path: Path) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], bbox_inches='tight', transparent=True, dpi=DPI / SCALE)


def print_im(n, im: np.ndarray) -> None:
    fig = plt.Figure(figsize=(SCALE, SCALE))
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.imshow(im, cmap='gray', interpolation='nearest')
    save_fig(fig, Path(f'../output/prediction_{n:06d}.png'))


def create_frame(model: torch.nn.Module, suffix: int) -> None:
    weights = [w.detach().cpu().numpy() for i, w in model.named_parameters() if 'weight' in i][1:-1]

    fig = plt.Figure(figsize=(len(weights) * SCALE, SCALE))

    for i, w in enumerate(weights):
        ax = fig.add_subplot(1, len(weights), i + 1)
        ax.set_axis_off()
        ax.imshow(w, cmap='gray', interpolation='nearest')

    save_fig(fig, Path(f'../output/{suffix:04d}.png'))


def main() -> None:
    assert torch.cuda.is_available()

    device = torch.device('cuda')

    im = cv2.imread('../images/mona-lisa.png', cv2.IMREAD_GRAYSCALE)

    fig = plt.Figure(figsize=(SCALE, SCALE))
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.imshow(im, cmap='gray', interpolation='nearest')
    save_fig(fig, Path('../output/original.png'))

    y = torch.tensor([[i] for i in im.flatten()], dtype=torch.float64, device=device)

    shape = im.shape
    grid = np.mgrid[0:shape[0], 0:shape[1]]

    x = torch.hstack([
        torch.tensor([[i] for i in grid[0].flatten()], dtype=torch.float64, device=device),
        torch.tensor([[i] for i in grid[1].flatten()], dtype=torch.float64, device=device),
    ])

    logging.debug(x)
    logging.debug(y)

    layer = 200

    model = nn.Sequential()
    model.append(nn.Linear(2, layer, bias=True, dtype=torch.float64))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=torch.float64))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=torch.float64))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, layer, bias=True, dtype=torch.float64))
    model.append(nn.Tanh())
    model.append(nn.Linear(layer, 1, bias=True, dtype=torch.float64))
    logging.info(model)

    model.to(device)
    n = 100_000

    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1,
                                  max_iter=n,
                                  max_eval=n,
                                  history_size=50,
                                  tolerance_grad=1e-17,
                                  tolerance_change=5e-12,
                                  line_search_fn='strong_wolfe')

    mse = nn.MSELoss()

    with tqdm(total=n, position=0, leave=True) as pbar, logging_redirect_tqdm():

        def closure():
            if pbar.n < n:
                pbar.update(1)
            if pbar.n % 100 == 0:
                create_frame(model, pbar.n)
                print_im(pbar.n, model.forward(x).detach().cpu().numpy().reshape(shape))

            optimizer.zero_grad()
            p = model.forward(x)
            loss = mse(p, y)
            if pbar.n % 100 == 0:
                logging.debug(f'Loss: {loss.item():.12f}')
            loss.backward()
            return loss

        model.train()
        optimizer.step(closure)

    model.eval()

    with torch.no_grad():
        create_frame(model, n)
        print_im(n, model.forward(x).detach().cpu().numpy().reshape(shape))


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
