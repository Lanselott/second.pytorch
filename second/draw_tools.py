import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numba
from numba import jit

import scipy
import numpy as np
import pdb
import torch

from IPython import embed

# def show_figures()

def draw_boxes(boxes_dict, fig_name, ratio=5):
    '''
    boxes_dict: {[x,y,w,l,r]},
    '''
    fig = plt.figure()
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    axes_limit=200
    ax.set_xlim(0, 352)
    ax.set_ylim(-axes_limit, axes_limit)

    for box in boxes_dict:
        x, y, w, l, r = box[0] * ratio, -box[1] * ratio, box[3] * ratio / 2, box[4] * ratio / 2, box[6]
        ax.add_patch(
                patches.Rectangle(
                (x, y),
                w,
                l,
                -r*180/np.pi,
                # edgecolor=edgecolor,
                fill=False      # remove background
            )) 
        ax.add_patch(
                patches.Rectangle(
                (x, y),
                l,
                w,
                -r*180/np.pi + 90,
                # edgecolor=edgecolor,
                fill=False      # remove background
            ))
        ax.add_patch(
                patches.Rectangle(
                (x, y),
                w,
                l,
                -r*180/np.pi + 180,
                # edgecolor=edgecolor,
                fill=False      # remove background
            ))
        ax.add_patch(
                patches.Rectangle(
                (x, y),
                l,
                w,
                -r*180/np.pi + 270,
                # edgecolor=edgecolor,
                fill=False      # remove background
            ))

    plt.savefig(fig_name)
