#!/usr/bin/env python3
"""Various deep models for pose prediction."""

from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, BatchNormalization, merge
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import h5py
import numpy as np

from common import huber_loss, unmap_predictions, convert_2d_seq, pck, THRESHOLDS

np.random.seed(2372143511)

# OFFSETS = [30, 60, 90, 120, 150]
OFFSETS = [30]
TAP_SUBSAMPLE = 5
TAP_COUNT = 10


def train_model(train_X, train_Y, val_X, val_Y):
    in_size = 2 * 8 * TAP_COUNT
    out_size = 2 * 8 * len(OFFSETS)

    in_layer = Input(shape=(in_size,))

    b0 = BatchNormalization()(in_layer)

    d1 = Dense(128)(b0)
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

    # huber works best so far; mae alone okay, but mse alone tends to overfit
    # objective = 'mae'
    # objective = 'mse'
    objective = huber_loss
    model.compile(optimizer='rmsprop', loss=objective, metrics=[objective])

    print('Fitting to data')
    mod_check = ModelCheckpoint('./best-weights.h5', save_best_only=True)
    estop = EarlyStopping(min_delta=0, patience=25)
    model.fit(train_X,
              train_Y,
              batch_size=256,
              validation_data=(val_X, val_Y),
              nb_epoch=2000,
              shuffle=True,
              callbacks=[mod_check, estop])

    return model


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
        # Originally I defined tap_ends to have a skip which would stop poses
        # from overlapping. That turned out to be a mildly harmful idea, so now
        # I'm using this approach instead.
        tap_ends = np.arange(TAP_SUBSAMPLE * (TAP_COUNT-1), last_tap).reshape((-1, 1))
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
    gt = unmap_predictions(val_Y, n=len(OFFSETS))
    val_preds = unmap_predictions(model.predict(val_X), n=len(OFFSETS))
    extend_preds = unmap_predictions(np.repeat(val_X[:, -16:], len(OFFSETS),
                                               axis=1),
                                     n=len(OFFSETS))
    np.save('val_pred.npy', val_preds)
    np.save('val_ext.npy', extend_preds)
    np.save('val_gt.npy', gt)

    pred_pcks = pck(gt, val_preds, THRESHOLDS, OFFSETS)
    ext_pcks = pck(gt, extend_preds, THRESHOLDS, OFFSETS)
    assert pred_pcks.keys() == ext_pcks.keys()
    print('metric\t\t\tpred\t\t\text')
    for k in pred_pcks.keys():
        print('{}\t\t{:g}\t\t{:g}'.format(k, pred_pcks[k], ext_pcks[k]))

# TODOs:
# - Can standardise input features more, potentially. Standardising
#   subsequences might be a very good idea.
# - Use a form of curriculum learning: Gaussian noise is slowly ramped up as
#   the system converges. In SRNN paper, this makes recurrent model less
#   susceptible to positive feedback loops when generating motion.
