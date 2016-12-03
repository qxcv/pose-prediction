#!/usr/bin/env python3

"""Various deep models for pose prediction."""

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
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
PA = [0, 0, 1, 2, 3, 1, 5, 6]


def convert_2d_seq(seq):
    """Convert a T*2*8 sequence of poses into a representation more amenable
    for learning. Returns T*F array of features."""
    assert seq.ndim == 3 and seq.shape[1:] == (2, 8)
    rv = seq.copy()

    # Begin by standardising data. Should (approximately) center the person
    rv = (rv - rv.mean()) / rv.std()

    # For non-root nodes, use distance-to-parent only
    # Results in root node (and root node only) storing absolute position
    for j, p in enumerate(PA):
        if j != p:
            rv[:, :, j] = seq[:, :, j] - seq[:, :, p]

    # Track velocity with head, instead of absolute position.
    rv[1:, :, 0] = rv[1:, :, 0] - rv[:-1, :, 0]
    rv[0, :, 0] = [0, 0]

    return rv.reshape((rv.shape[0], -1))


# def unmap_predictions(seq):
#     """Convert predictions back into actual poses. Assumes that original
#     sequence (including input) was processed with `convert_2d_seq`. Always puts
#     position of first predicted head at [0, 0]. Subsequent poses have head
#     positions defined by previous ones."""
#     seq.reshape
#     rv = zeros()


def pck_metric(threshs, joints=None, offsets=OFFSETS):
    def inner(y_true, y_pred):
        nt = len(offsets)
        rv = {}

        for thresh in threshs:
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


def prepare_data(fp, val_frac=0.2):
    data = []
    labels = []

    # These are used to offset from the tap_ends array
    in_offsets = np.arange(-TAP_SUBSAMPLE * (TAP_COUNT - 1), 1, TAP_SUBSAMPLE) \
                   .reshape(1, -1)
    out_offsets = np.array(OFFSETS).reshape((1, -1))
    sizes = []

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

        sizes.append(in_data.shape[0])

    sizes = np.array(sizes)
    data = np.array(data)
    labels = np.array(labels)

    # Do train/val split
    required_val = val_frac * sum(sizes)
    indices = np.arange(len(sizes))
    np.random.shuffle(indices)
    cum_sizes = np.cumsum(sizes[indices])
    ind_is_val = cum_sizes < required_val
    ind_is_train = ~ind_is_val
    val_inds = indices[ind_is_val]
    train_inds = indices[ind_is_train]

    train_X = data[train_inds]
    train_Y = labels[train_inds]
    val_X = data[val_inds]
    val_Y = labels[val_inds]

    return tuple(map(np.concatenate, [train_X, train_Y, val_X, val_Y]))


def train_model(train_X, train_Y, val_X, val_Y):
    in_size = 2 * 8 * TAP_COUNT
    out_size = 2 * 8 * len(OFFSETS)
    model = Sequential([
        Dense(128, input_dim=in_size),
        Activation('relu'),
        BatchNormalization(),
        Dense(128),
        Activation('relu'),
        BatchNormalization(),
        Dense(128),
        Activation('relu'),
        BatchNormalization(),
        Dense(128),
        Activation('relu'),
        Dense(out_size),
    ])
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    print('Fitting to data')
    mod_check = ModelCheckpoint('./best-weights.h5', save_best_only=True)
    model.fit(train_X, train_Y, batch_size=1024, validation_data=(val_X, val_Y),
            nb_epoch=1000, shuffle=True, callbacks=[mod_check])
    return model


if __name__ == '__main__':
    print('Loading data')
    with h5py.File('h36m-poses.h5', 'r') as fp:
        train_X, train_Y, val_X, val_Y = prepare_data(fp)
    print('Data loaded')

    try:
        print('Loading model')
        model = load_model('./best-weights.h5')
    except OSError:
        print('Load failed, building model anew')
        model = train_model(train_X, train_Y, val_X, val_Y)

    # TODO: Check validation loss with PCK
