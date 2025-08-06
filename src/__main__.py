import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import matplotlib.pyplot as plt

DPI: int = 1000
SCALE: float = 2.5


def save_fig(fig: plt.Figure, path: Path) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], bbox_inches='tight', transparent=True, dpi=DPI / SCALE)


def create_frame(model: torch.nn.Module, suffix: int) -> None:
    weights = [w.detach().cpu().numpy() for w in model.parameters()]

    fig = plt.Figure(figsize=(len(weights) * SCALE, SCALE))

    for i, w in enumerate(weights):
        ax = fig.add_subplot(1, len(weights), i + 1)
        ax.set_axis_off()
        ax.imshow(w, cmap='gray', interpolation='nearest')

    save_fig(fig, Path(f'../output/{suffix:04d}.png'))


def main() -> None:
    assert torch.cuda.is_available()

    device = torch.device('cuda')

    x = [0, 1, 2, 3, 4]
    y = [9, 8, 7, 6, 5]

    __x = torch.tensor(x, dtype=torch.float64, device=device)
    __y = torch.tensor(y, dtype=torch.float64, device=device)

    model = nn.Sequential()
    model.append(nn.Linear(__x.size(dim=0), 10, bias=False, dtype=torch.float64))
    model.append(nn.Tanh())
    model.append(nn.Linear(10, 10, bias=False, dtype=torch.float64))
    model.append(nn.Tanh())
    model.append(nn.Linear(10, __y.size(dim=0), bias=False, dtype=torch.float64))

    model.to(device)

    print(model)

    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1,
                                  max_iter=100,
                                  max_eval=100,
                                  history_size=50,
                                  tolerance_grad=1e-17,
                                  tolerance_change=5e-12,
                                  line_search_fn='strong_wolfe')

    mse = nn.MSELoss()

    with tqdm(total=100, position=0, leave=True) as pbar, logging_redirect_tqdm():

        def closure():
            if pbar.n < 100:
                pbar.update(1)
            create_frame(model, pbar.n)

            optimizer.zero_grad()
            p = model.forward(__x)
            loss = mse(p, __y)
            logging.debug(p.detach().cpu().tolist()[0])
            loss.backward()
            return loss

        model.train()
        optimizer.step(closure)

    create_frame(model, 9999)

    model.eval()
    e = model.forward(__x).detach().cpu().tolist()
    logging.info(e)


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
