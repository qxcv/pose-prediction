#!/usr/bin/env python3
"""Various deep models for pose prediction."""

from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, BatchNormalization, merge
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import numpy as np

np.random.seed(2372143511)

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

    # Begin by standardising data. Should (approximately) center the person,
    # without distorting width and height.
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


def unmap_predictions(seq):
    """Convert predictions back into actual poses. Assumes that original
    sequence (including input) was processed with `convert_2d_seq`. Always puts
    position of first predicted head at [0, 0]. Subsequent poses have head
    positions defined by previous ones.

    Returns an array which is (no. samples)*(no. pred offsets)*2*8."""
    seq = seq.reshape((-1, len(OFFSETS), 2, 8))
    rv = np.zeros_like(seq)

    # Undo offset-based nonsense for everything below the head (puts the head
    # at zero)
    for joint in range(1, 8):
        parent = PA[joint]
        rv[:, :, :, joint] = seq[:, :, :, joint] + rv[:, :, :, parent]

    # Undo head motion (assumes head at zero in final frame)
    rv[:, :, :, 0] = seq[:, :, :, 0]
    for time in range(1, len(OFFSETS)):
        delta = seq[:, time, :, 0:1]
        offset = delta + rv[:, time - 1, :, 0:1]
        rv[:, time, :, :] += offset

    return rv


def pckh_metric(threshs, joints=None, offsets=OFFSETS):
    """Calculate head-neck normalised PCK."""

    # Known bugs:
    # - Unclear whether reshape on y_true, y_pred is correct
    # - Code doesn't unmap predictions correctly, so this function won't work
    #   if used to produce a metric for Model.fit() (which is what it was
    #   intended for in the first place…)
    assert False, "Fix marked bugs first"

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
            heads = np.linalg.norm(pred_s[0, :, :] - pred_s[1, :, :])
            pck = (dists < heads * thresh).sum()
            for off_i, off in enumerate(offsets):
                label = 'pckh@%.2f/%d' % (thresh, off)
                rv[label] = pck[off_i]

        return rv

    return inner


def prepare_data(fp, val_frac=0.2):
    data = []
    labels = []

    # These are used to offset from the tap_ends array
    in_offsets = np.arange(-TAP_SUBSAMPLE * (TAP_COUNT-1), 1, TAP_SUBSAMPLE) \
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
        tap_ends = np.arange(TAP_SUBSAMPLE * (TAP_COUNT-1), last_tap,
                             TAP_SUBSAMPLE * TAP_COUNT).reshape((-1, 1))
        in_indices = tap_ends + in_offsets
        out_indices = tap_ends + out_offsets

        converted = convert_2d_seq(poses)
        in_data = converted[in_indices]
        shaped_in = in_data.reshape((in_data.shape[0], -1))
        data.append(shaped_in)
        out_data = converted[out_indices]
        shaped_out = out_data.reshape((out_data.shape[0], -1))
        labels.append(shaped_out)

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

    in_layer = Input(shape=(in_size, ))

    d1 = Dense(128)(in_layer)
    a1 = Activation('relu')(d1)
    b1 = BatchNormalization()(a1)

    d2 = Dense(128)(b1)
    a2 = Activation('relu')(d2)
    b2 = BatchNormalization()(a2)

    m2 = merge([b1, b2], mode='sum')

    d3 = Dense(128)(m2)
    a3 = Activation('relu')(d3)
    b3 = BatchNormalization()(a3)

    m3 = merge([b3, m2], mode='sum')

    d4 = Dense(128)(m3)
    a4 = Activation('relu')(d4)
    b4 = BatchNormalization()(a4)

    m4 = merge([b4, m3], mode='sum')

    out_layer = Dense(out_size)(m4)

    model = Model(input=[in_layer], output=[out_layer])

    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])

    print('Fitting to data')
    mod_check = ModelCheckpoint('./best-weights.h5', save_best_only=True)
    estop = EarlyStopping(min_delta=0, patience=25)
    model.fit(train_X,
              train_Y,
              batch_size=1024,
              validation_data=(val_X, val_Y),
              nb_epoch=2000,
              shuffle=True,
              callbacks=[mod_check, estop])

    return model


def load_data():
    with h5py.File('h36m-poses.h5', 'r') as fp:
        return prepare_data(fp)


if __name__ == '__main__':
    print('Loading data')
    train_X, train_Y, val_X, val_Y = load_data()
    print('Data loaded')

    model = None
    try:
        print('Loading model')
        model = load_model('./best-weights.h5')
    except OSError:
        print('Load failed, building model anew')
    if model is None:
        model = train_model(train_X, train_Y, val_X, val_Y)

    # Calculate PCK and MAE on val set, comparing with effective-but-stupid
    # "extend" base line
    print('Checking prediction accuracy')
    gt = unmap_predictions(val_Y)
    val_preds = unmap_predictions(model.predict(val_X))
    extend_preds = unmap_predictions(
        np.repeat(
            val_X[:, -16:], len(OFFSETS), axis=1))
    np.save('val_pred.npy', val_preds)
    np.save('val_ext.npy', val_preds)
    np.save('val_gt.npy', gt)

# TODOs:
# - Can standardise input features more, potentially. Standardising
#   subsequences might be a very good idea.
# - Use a form of curriculum learning: Gaussian noise is slowly ramped up as
#   the system converges. In SRNN paper, this makes recurrent model less
#   susceptible to positive feedback loops when generating motion.
