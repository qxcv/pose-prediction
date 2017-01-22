#!/usr/bin/env python3

"""Uses encoder to find feature vectors for H36M data, then trains a linear
classifier on the feature vectors. May end up with waaaay too many dimensions
to generalise meaningfully; we'll see."""

from glob import glob
from os import path
from multiprocessing import Pool
import re

from keras.models import Model, load_model
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import numpy as np

from common import GOOD_MOCAP_INDS

SEQ_SKIP = 3
SEQ_LENGTH = 32
MODEL_PATH = './seq-vae/models/epoch-200-enc.h5'
ACTION_IDS = {
    'eating': 0,
    'walking': 1,
    'smoking': 2,
}

np.random.seed(2372143511)


def make_truncated(seq_length, pose_size):
    x = in_layer = Input(shape=(seq_length, pose_size,))
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = LSTM(1000, return_sequences=False)(x)
    x = out_layer = Dense(500)(x)
    # No mean/var layers

    return Model(input=[in_layer], output=[out_layer])


def convert_to_truncated(model):
    # Create new model mimicking old one, but with stateful LSTM. Not sure how
    # to do this for general input models.
    _, model_steps, pose_size = model.input_shape
    new_model = make_truncated(SEQ_LENGTH, pose_size)
    for src_layer, dest_layer in zip(model.layers, new_model.layers):
        dest_layer.set_weights(src_layer.get_weights())
    return new_model


def get_encoder():
    # Use a thread-local in case this is run in a multi-threaded setup
    orig_model = load_model(MODEL_PATH)
    return convert_to_truncated(orig_model)


def prepare_file(filename, seq_length, seq_skip):
    poses = np.loadtxt(filename, delimiter=',')
    assert poses.ndim == 2 and poses.shape[1] == 99, poses.shape

    zero_inds, = np.nonzero((poses != 0).any(axis=0))
    assert (zero_inds == GOOD_MOCAP_INDS).all(), zero_inds
    poses = poses[:, GOOD_MOCAP_INDS]

    seqs = []
    true_length = seq_length * seq_skip
    end = len(poses) - true_length + 1
    # TODO: Might not want to overlap sequences so much. Then again, it may not
    # matter given that I'm shuffling anyway
    for start in range(end):
        seqs.append(poses[start:start+true_length:seq_skip])

    return np.stack(seqs)


def is_valid(data):
    return np.isfinite(data).all()


_fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')
def _mapper(arg):
    """Worker to load data in parallel"""
    filename, seq_length, seq_skip = arg
    base = path.basename(filename)
    meta = _fnre.match(base).groupdict()
    subj_id = int(meta['subject'])
    X = prepare_file(filename, seq_length, seq_skip)
    for act_key in ACTION_IDS:
        if act_key in filename:
            act_id = ACTION_IDS[act_key]
            Y = act_id * np.ones((len(X),))
            break

    return subj_id, X, Y


def load_data(seq_length=SEQ_LENGTH, seq_skip=SEQ_SKIP):
    filenames = glob('h36m-3d-poses/expmap_*.txt.gz')

    train_X_blocks = []
    train_Y_blocks = []
    test_X_blocks = []
    test_Y_blocks = []

    print('Spawning pool')
    with Pool() as pool:
        fn_seq = ((fn, seq_length, seq_skip) for fn in filenames)
        for subj_id, X, Y in pool.map(_mapper, fn_seq):
            if subj_id == 5:
                # subject 5 is for testing
                test_X_blocks.append(X)
                test_Y_blocks.append(Y)
            else:
                train_X_blocks.append(X)
                train_Y_blocks.append(Y)

    train_Y = np.concatenate(train_Y_blocks, axis=0)
    test_Y = np.concatenate(test_Y_blocks, axis=0)

    # Memory usage is right on the edge of what small machines are capable of
    # handling here, so I'm being careful to delete large unneeded structures.
    train_X = np.concatenate(train_X_blocks, axis=0)
    del train_X_blocks
    test_X = np.concatenate(test_X_blocks, axis=0)
    del test_X_blocks

    N, T, D = train_X.shape

    reshaped = train_X.reshape((N*T, D))
    mean = reshaped.mean(axis=0).reshape((1, 1, -1))
    std = reshaped.std(axis=0).reshape((1, 1, -1))
    del reshaped

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    assert is_valid(train_X)
    assert is_valid(train_Y)
    assert is_valid(test_X)
    assert is_valid(test_Y)

    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    print('Loading data')
    train_pose, train_Y, val_pose, val_Y = load_data(SEQ_LENGTH, SEQ_SKIP)
    print('Data loaded')

    print('Loading model')
    encoder = get_encoder()
    print('Getting train predictions')
    train_X = encoder.predict(train_pose, verbose=1, batch_size=128)
    print('Getting validation predictions')
    val_X = encoder.predict(val_pose, verbose=1, batch_size=128)

    print('Training SVC')
    model = LinearSVC(C=1e-6)
    model.fit(train_X, train_Y)
    target_names = sorted(ACTION_IDS.keys(), key=lambda a: ACTION_IDS[a])
    print('Evaluating SVC on training set')
    train_out_Y = model.predict(train_X)
    print(classification_report(train_Y, train_out_Y, target_names=target_names))

    print('Evaluating SVC on validation set')
    val_out_Y = model.predict(val_X)
    print(classification_report(val_Y, val_out_Y, target_names=target_names))
