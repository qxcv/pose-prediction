#!/usr/bin/env python3

"""Encoder-recurrent-decoder architecture for forecasting.

Not the same as the original ERD model for forecasting! They took images as
input and gave forecasted poses as output. This model takes in pose maps as
input and gives forecasted poses (at time t+K) as output."""

from keras.models import Model, load_model
from keras.layers import Input, Activation, TimeDistributed, \
    BatchNormalization, Convolution2D, ConvLSTM2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import numpy as np

from common import CUSTOM_OBJECTS, heatmapify_batch

np.random.seed(2372143511)

# Will result in learning mappings from sequences of SEQ_LENGTH - 1 to
# SEQ_LENGTH - 1
SEQ_LENGTH = 129
FRAME_SIZE = 32
WEIGHTS = './best-erd-weights.h5'


def make_model_train(shape):
    timesteps, rows, cols, njoints = shape

    # TODO: These channel counts are probably quite a bit too low. Need to bump
    # a bit.

    # Input layer (BN just for scaling)
    x = in_layer = Input(shape=(timesteps, rows, cols, njoints))
    x = TimeDistributed(BatchNormalization())(x)

    # Encoder
    for i in range(3):
        x = TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'))(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(BatchNormalization())(x)

    # Recurrent
    x = ConvLSTM2D(64, 3, 3, border_mode='same', return_sequences=True)(x)

    # Decoder
    x = TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Activation('relu'))(x)
    out_layer = x = TimeDistributed(Convolution2D(njoints, 3, 3, border_mode='same'))(x)

    model = Model(input=[in_layer], output=[out_layer])

    return model


def train_model(train_gen, val_gen, data_shape):
    model = make_model_train(data_shape)

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

    print('Fitting to data')
    mod_check = ModelCheckpoint(WEIGHTS, save_best_only=True)
    estop = EarlyStopping(min_delta=0, patience=250)
    callbacks = [
        mod_check, estop
    ]
    model.fit_generator(train_gen,
                        samples_per_epoch=10000,
                        nb_epoch=2000,
                        validation_data=val_gen,
                        nb_val_samples=100,
                        callbacks=callbacks)

    return model


def box_seq(seq, side=FRAME_SIZE, margin=3):
    """Rescale sequence to be side*side (with margin containing no joints)."""
    # XXX: This is going to cause a lot of camera movement between frames. Is
    # it really a good idea?
    mins = seq.min(axis=2)
    maxs = seq.max(axis=2)
    # scale for each frame is size of box required to contain pose
    scales = np.abs(maxs - mins).max(axis=1)
    mids = (maxs + mins) / 2.0
    seq -= mids.reshape(mids.shape + (1,))
    seq /= scales.reshape(scales.shape + (1, 1))
    seq += 0.5
    # now everything is in [0, 1], so we can put it into [margin, side-margin]
    seq = (side - 2 * margin) * seq + margin
    return seq


def _data_generator(data, batch_size=1):
    while True:
        ordering = np.random.permutation(len(data))
        for start in range(0, len(data) - batch_size + 1, batch_size):
            inds = ordering[start:start+batch_size]
            batch_poses = data[inds]
            batch_heatmaps = heatmapify_batch(batch_poses, (FRAME_SIZE, FRAME_SIZE))
            yield (batch_heatmaps[:, :-1], batch_heatmaps[:, 1:])


def prepare_data(fp, val_frac=0.2):
    data = []

    for ds_name in fp.keys():
        if not ds_name.startswith('poses_'):
            continue
        ds = fp[ds_name]
        poses = ds.value
        # split the poses into groups of SEQ_LENGTH, then box them individually
        for start in range(0, len(poses) - SEQ_LENGTH + 1, SEQ_LENGTH):
            subseq = poses[start:start+SEQ_LENGTH]
            boxed = box_seq(subseq)
            data.append(boxed)

    data = np.stack(data)

    # Train/val split
    required_val = int(val_frac * len(data))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    val_inds = indices[:required_val]
    train_inds = indices[required_val:]

    train_gen = _data_generator(data[train_inds])
    val_gen = _data_generator(data[val_inds])

    return train_gen, val_gen


def load_data():
    with h5py.File('h36m-poses.h5', 'r') as fp:
        return prepare_data(fp)


if __name__ == '__main__':
    print('Loading data')
    train_gen, val_gen = load_data()
    print('Data loaded')

    assert K.image_dim_ordering() == 'tf', \
        "Expecting TensorFlow dim ordering (even with Theano back end)"

    model = None
    try:
        print('Loading model')
        model = load_model(WEIGHTS, CUSTOM_OBJECTS)
    except OSError:
        print('Load failed, building model anew')
    if model is None:
        # TODO: Get data_shape in a more sane way
        data_shape = (SEQ_LENGTH - 1, FRAME_SIZE, FRAME_SIZE, 8)
        model = train_model(train_gen, val_gen, data_shape)
