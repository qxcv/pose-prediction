#!/usr/bin/env python3

"""Grab a random pose out of a .npy file and plot it. Meant to work in tandem
with predict_pose.py; ignore this if it doesn't make sense anymore."""

import matplotlib.pyplot as plt
import numpy as np

PA = [0, 0, 1, 2, 3, 1, 5, 6]
TYPES = ['ext', 'gt', 'pred']


def plot_pose(pose, c='r', old_h=None):
    """Plot a pose, or update an old pose"""
    if old_h is None:
        h = []
    else:
        h = old_h
    for j in range(len(PA)):
        x1, y1 = pose[:, j]
        x2, y2 = pose[:, PA[j]]
        data = ([x1, x2], [y1, y2])
        if old_h is None:
            a, = plt.plot(*data, color=c)
            h.append(a)
        else:
            h[j].set_data(*data)
    return h


if __name__ == '__main__':
    data = np.load('val_gt.npy')
    chosen_k = np.random.randint(data.shape[0])
    chosen_t = np.random.randint(data.shape[1])
    minima = np.inf * np.ones(2)
    maxima = -np.inf * np.ones(2)
    handles = []
    labels = []
    for t, c in [('gt', 'g'), ('pred', 'r'), ('ext', 'b')]:
        data = np.load('val_' + t + '.npy')
        pose = data[chosen_k, chosen_t]
        h = plot_pose(pose)[0]
        # Only add last handle (w/ right colour) to make legend
        handles.append(h)
        labels.append(t)
        # Add 10% margin on either side of pose
        minima = np.minimum(pose.min(axis=1), minima)
        maxima = np.maximum(pose.max(axis=1), maxima)
    assert maxima.shape == (2,)
    assert minima.shape == (2,)
    extra = 0.5 * max(abs(maxima - minima))
    maxima += extra
    minima -= extra
    plt.axis('equal')
    plt.xlim(minima[0], maxima[0])
    plt.ylim(minima[1], maxima[1])
    plt.gca().invert_yaxis()
    plt.legend(handles, labels)
    plt.show()
