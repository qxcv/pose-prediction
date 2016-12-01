#!/usr/bin/env python3
"""Various deep models for pose prediction."""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import h5py
import numpy as np

# Idea: predict at OFFSETS[0] after the last input frame, OFFSETS[1] after
# the last input frame, etc.
# TODO: SRNN paper suggests several things. Should try at least some.
# - They use an angular, tree-based parameterisation (exponential map based on
#   quaternions) to represent joint positions. May be more amenable to motion
#   modelling.
# - They always standardise input features (not sure about output features?)
# - They use a form of curriculum learning: Gaussian noise is slowly ramped up
#   as the system converges. This makes it less susceptible to positive
#   feedback loops when generating motion.
# OFFSETS = [30, 60, 90, 120, 150]
OFFSETS = [30]
TAP_SUBSAMPLE = 5
TAP_COUNT = 5


def convert_2d_seq(seq):
    """Convert a T*2*8 sequence of poses into a representation more amenable
    for learning. Returns T*F array of features."""
    assert seq.ndim == 3 and seq.shape[1:] == (2, 8)
    pa = [0, 0, 1, 2, 3, 1, 5, 6]
    rv = seq.copy()
    for j, p in enumerate(pa):
        if j != p:
            # For non-root nodes, use distance-to-parent only
            # Results in root node (and root node only) storing absolute
            # position
            rv[:, :, j] = seq[:, :, j] - seq[:, :, p]
    return rv.reshape((rv.shape[0], -1))


def pck_metric(threshs, joints=None, offsets=OFFSETS):
    def inner(y_true, y_pred):
        nt = len(offsets)
        rv = {}

        for tresh in threshs:
            true_s = np.reshape(y_true, [8, 2, nt])
            pred_s = np.reshape(y_pred, [8, 2, nt])
            if joints is not None:
                true_s = true_s[joints]
                pred_s = pred_s[joints]
            dists = np.linalg.norm(true_s - pred_s, axis=1)
            pck = (dists < thresh).sum()
            for off_i, off in enumerate(offsets):
                label = 'pck@%.2f/%d' % (thresh, off)
                rv[label] = pck[off_i]

        return rv

    return inner


def prepare_data(fp):
    data = []
    labels = []

    # These are used to offset from the tap_ends array
    in_offsets = np.arange(-TAP_SUBSAMPLE * (TAP_COUNT - 1), 1, TAP_SUBSAMPLE) \
                   .reshape(1, -1)
    out_offsets = np.array(OFFSETS).reshape((1, -1))

    for ds_name in fp.keys():
        if not ds_name.startswith('poses_'):
            continue
        ds = fp[ds_name]
        poses = ds.value

        # Figure out an n*<dim> matrix for both input data and output data
        last_tap = poses.shape[0] - 1 - max(OFFSETS)
        tap_ends = np.arange(TAP_SUBSAMPLE * (TAP_COUNT-1), last_tap, TAP_SUBSAMPLE * TAP_COUNT)\
                     .reshape((-1, 1))
        in_indices = tap_ends + in_offsets
        out_indices = tap_ends + out_offsets

        converted = convert_2d_seq(poses)
        in_data = converted[in_indices]
        data.append(in_data.reshape((in_data.shape[0], -1)))
        out_data = converted[out_indices]
        labels.append(out_data.reshape((out_data.shape[0], -1)))

    return np.concatenate(data), np.concatenate(labels)


if __name__ == '__main__':
    print('Loading data')
    with h5py.File('h36m-poses.h5', 'r') as fp:
        in_data, labels = prepare_data(fp)
    print('Data loaded')
    in_size = 2 * 8 * TAP_COUNT
    out_size = 2 * 8 * len(OFFSETS)
    print('Building model')
    model = Sequential([
        Dense(
            128, input_dim=in_size),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(out_size),
    ])
    model.compile(optimizer='rmsprop', loss='mae')
    print('Fitting to data')
    model.fit(in_data,
              labels,
              batch_size=2048,
              nb_epoch=100,
              validation_split=0.2,
              shuffle=True)
