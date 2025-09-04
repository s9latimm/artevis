from time import sleep

import cv2
import numpy as np
from matplotlib import pyplot as plt, colors
import matplotlib.animation as anim
from matplotlib.patches import Rectangle
from tqdm import tqdm

from src import OUTPUT_DIR, SEISMIC
from src.artevis.__main__ import save_fig, image


def main():
    # plt.ion()
    dpi = 200
    fig = plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

    nodes = np.array([
        (0, 1.5),
        (0, -.5),
        (2, 3),
        (2, 2),
        (2, 1),
        (2, 0),
        (2, -1),
        (2, -2),
        (4, 3),
        (4, 2),
        (4, 1),
        (4, 0),
        (4, -1),
        (4, -2),
        (6, 3),
        (6, 2),
        (6, 1),
        (6, 0),
        (6, -1),
        (6, -2),
        (8, 3),
        (8, 2),
        (8, 1),
        (8, 0),
        (8, -1),
        (8, -2),
        (10, 2),
        (10, .5),
        (10, -1),
    ])

    np.random.seed(48)
    w1_0 = np.random.random((6, 6)) * 2 - 1
    w2_0 = np.random.random((6, 6)) * 2 - 1
    w3_0 = np.random.random((6, 6)) * 2 - 1

    w1_0 **= 7
    w2_0 **= 7
    w3_0 **= 7

    # w3 = np.rot90(w3)
    # w2 = np.rot90(w2)
    # w1 = np.rot90(w1)

    orig = image('fog', 8)[::-1, :, ::-1]
    im = orig.copy()
    art = im.copy()

    # path = OUTPUT_DIR / 'fog/0-0-0-1-1_art.jpg'
    # im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # im = cv2.resize(im, (round(8 / im.shape[0] * im.shape[1]), 8))[::-1, :, ::-1]
    #
    # path = OUTPUT_DIR / 'fog/0-0-1-1-1_art.jpg'
    # art = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # art = cv2.resize(art, (round(8 / im.shape[0] * im.shape[1]), 8))[::-1, :, ::-1]

    frame = 1
    for y_off in range(im.shape[0]):
        for x_off in range(im.shape[1]):

            w1 = w1_0 - ((48 - frame) / 48) * (np.random.random(
                (6, 6)) * 2 - 1)**(y_off + 1 if y_off % 2 == 0 else y_off)
            w2 = w2_0 - ((48 - frame) / 48) * (np.random.random(
                (6, 6)) * 2 - 1)**(y_off + 1 if y_off % 2 == 0 else y_off)
            w3 = w3_0 - ((48 - frame) / 48) * (np.random.random(
                (6, 6)) * 2 - 1)**(y_off + 1 if y_off % 2 == 0 else y_off)

            ax = fig.add_subplot(111)
            ax.axis('equal')
            fig.subplots_adjust(bottom=0.05, top=.95, left=.05, right=.95)
            ax.margins(0, 0)
            ax.set_axis_off()

            ax.scatter(nodes[:, 0], nodes[:, 1], color='k', clip_on=False, zorder=999)
            for a in range(-2, 4):
                ax.plot([0, 2], [-.5, a], lw=1, color='k', zorder=3, clip_on=False)
                ax.plot([0, 2], [1.5, a], lw=1, color='k', zorder=3, clip_on=False)
                ax.plot([8, 10], [a, 2], lw=1, color='k', zorder=3, clip_on=False)
                ax.plot([8, 10], [a, .5], lw=1, color='k', zorder=3, clip_on=False)
                ax.plot([8, 10], [a, -1], lw=1, color='k', zorder=3, clip_on=False)
                for b in range(-2, 4):
                    ax.plot([2, 4], [a, b], lw=1, color='k', zorder=3, clip_on=False)
                    ax.plot([4, 6], [a, b], lw=1, color='k', zorder=3, clip_on=False)
                    ax.plot([6, 8], [a, b], lw=1, color='k', zorder=3, clip_on=False)

            print(im.shape)

            y, x = np.mgrid[0:im.shape[0], -im.shape[1]:0]
            x = x.astype(np.float64) - 1.5
            y = y.astype(np.float64) - 2
            ax.pcolor(x, y, orig)

            y, x = np.mgrid[0:im.shape[0], 0:im.shape[1]]
            x = x.astype(np.float64) + 12.5
            y = y.astype(np.float64) - 2
            res = im.copy()
            res = res * 0 + 255

            # for y_im in range(y_off):
            #     for x_im in range(6):
            #         res[-1 - y_im][x_im] = art[-1 - y_im][x_im]
            #
            # for x_im in range(x_off + 1):
            #     res[-1 - y_off][x_im] = art[-1 - y_off][x_im]

            ax.pcolor(x, y, res)

            ax.plot([-im.shape[1] - 1 + x_off, 0], [im.shape[0] - 3.5 - y_off, -.5],
                    '--',
                    lw=1,
                    color='r',
                    zorder=4,
                    clip_on=False)
            ax.plot([-im.shape[1] - 1 + x_off, 0], [im.shape[0] - 2.5 - y_off, 1.5],
                    '--',
                    lw=1,
                    color='r',
                    zorder=4,
                    clip_on=False)
            ax.add_patch(
                Rectangle((-im.shape[1] - 2 + x_off, im.shape[0] - 3.5 - y_off),
                          1,
                          1,
                          color='r',
                          lw=1,
                          fill=False,
                          zorder=9999,
                          clip_on=False))

            if False:
                ax.plot([12 + x_off, 10], [im.shape[0] - 2.5 - y_off, 2],
                        '--',
                        lw=1,
                        color='r',
                        zorder=4,
                        clip_on=False)
                ax.plot([12 + x_off, 10], [im.shape[0] - 3 - y_off, .5], '--', lw=1, color='g', zorder=4, clip_on=False)
                ax.plot([12 + x_off, 10], [im.shape[0] - 3.5 - y_off, -1],
                        '--',
                        lw=1,
                        color='b',
                        zorder=4,
                        clip_on=False)

                ax.add_patch(
                    Rectangle((12 + x_off, im.shape[0] - 3.5 - y_off),
                              1,
                              1,
                              color='r',
                              lw=1,
                              fill=False,
                              zorder=9999,
                              clip_on=False))

            ax.add_patch(
                Rectangle((-8, -2.5), im.shape[1], im.shape[0], color='gray', lw=1, fill=False, zorder=3,
                          clip_on=False))
            ax.add_patch(
                Rectangle((12, -2.5), im.shape[1], im.shape[0], color='gray', lw=1, fill=False, zorder=3,
                          clip_on=False))

            y, x = np.mgrid[0:6, 0:6]
            x = x.astype(np.float64) - 8 + 2.5
            y = y.astype(np.float64) - 11.5

            norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            cmap = SEISMIC

            ax.pcolor(x, y, w1, norm=norm, cmap=cmap)

            ax.plot([2, -6], [-2, -6], '--', lw=1, color='gray', zorder=4, clip_on=False)
            ax.plot([4, 0], [-2, -6], '--', lw=1, color='gray', zorder=4, clip_on=False)
            ax.add_patch(Rectangle((-6, -12), 6, 6, color='gray', lw=1, fill=False, zorder=3, clip_on=False))

            y, x = np.mgrid[0:6, 0:6]
            x = x.astype(np.float64) + 2.5
            y = y.astype(np.float64) - 11.5
            ax.pcolor(x, y, w2, norm=norm, cmap=cmap)

            ax.plot([4, 2], [-2, -6], '--', lw=1, color='gray', zorder=4, clip_on=False)
            ax.plot([6, 8], [-2, -6], '--', lw=1, color='gray', zorder=4, clip_on=False)
            ax.add_patch(Rectangle((2, -12), 6, 6, color='gray', lw=1, fill=False, zorder=3, clip_on=False))

            y, x = np.mgrid[0:6, 0:6]
            x = x.astype(np.float64) + 8 + 2.5
            y = y.astype(np.float64) - 11.5
            ax.pcolor(x, y, w3, norm=norm, cmap=cmap)

            ax.plot([6, 10], [-2, -6], '--', lw=1, color='gray', zorder=4, clip_on=False)
            ax.plot([8, 16], [-2, -6], '--', lw=1, color='gray', zorder=4, clip_on=False)
            ax.add_patch(Rectangle((10, -12), 6, 6, color='gray', lw=1, fill=False, zorder=3, clip_on=False))

            save_fig(fig, OUTPUT_DIR / f"network/frames/frame_{frame:06d}.png", dpi)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.clf()

            frame += 1

    plt.close('all')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
