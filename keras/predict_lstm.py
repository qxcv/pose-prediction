#!/usr/bin/env python3

"""LSTM-based model for predicting poses, one frame at a time"""

from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, LSTM, TimeDistributed, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import numpy as np

from common import huber_loss, convert_2d_seq

np.random.seed(2372143511)

# Will result in learning mappings from sequences of SEQ_LENGTH - 1 to
# SEQ_LENGTH - 1
SEQ_LENGTH = 129

def train_model(train_X, train_Y, val_X, val_Y):
    assert train_X.ndim == 3
    step_data_size = train_X.shape[-1]

    x = in_layer = Input(shape=train_X.shape[1:])
    x = TimeDistributed(BatchNormalization())(x)
    x = LSTM(128, return_sequences=True)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(BatchNormalization())(x)
    out_layer = TimeDistributed(Dense(step_data_size))(x)

    model = Model(input=[in_layer], output=[out_layer])

    objective = huber_loss
    model.compile(optimizer='rmsprop', loss=objective, metrics=[objective])

    print('Fitting to data')
    mod_check = ModelCheckpoint('./best-lstm-weights.h5', save_best_only=True)
    estop = EarlyStopping(min_delta=0, patience=25)
    # TODO: Need curriculum learning for forecasting to work properly
    model.fit(train_X,
              train_Y,
              # batch_size=256,
              validation_data=(val_X, val_Y),
              nb_epoch=2000,
              shuffle=True,
              callbacks=[mod_check, estop])

    return model


def prepare_data(fp, val_frac=0.2):
    data = []

    for ds_name in fp.keys():
        if not ds_name.startswith('poses_'):
            continue
        ds = fp[ds_name]
        poses = ds.value

        # Ensure that all sequences are of the same length so that we don't
        # have to mask
        converted = convert_2d_seq(poses)
        seqs = []
        for start in range(0, len(converted) - SEQ_LENGTH + 1, SEQ_LENGTH):
            seqs.append(converted[start:start+SEQ_LENGTH])

        data.extend(seqs)

    data = np.stack(data)

    # Train/val split
    required_val = int(val_frac * len(data))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    val_inds = indices[:required_val]
    train_inds = indices[required_val:]

    X = data[:, :-1, :]
    Y = data[:, 1:, :]

    train_X = X[train_inds]
    train_Y = Y[train_inds]
    val_X = X[val_inds]
    val_Y = Y[val_inds]

    return train_X, train_Y, val_X, val_Y


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
        model = load_model('./best-lstm-weights.h5')
    except OSError:
        print('Load failed, building model anew')
    if model is None:
        model = train_model(train_X, train_Y, val_X, val_Y)

    # Compare against 'extend' baseline, as per usual
    # TODO: This is a poor approach; I really need to predict out to a few
    # frames and check PCK. Having a visualisation wouldn't hurt either
    # (although that can be done on my side).
    print('Checking prediction accuracy')
    # gt = unmap_predictions(val_Y, n=len(OFFSETS))
    # val_preds = unmap_predictions(model.predict(val_X), n=len(OFFSETS))
    # extend_preds = unmap_predictions(np.repeat(val_X[:, -16:], len(OFFSETS),
    #                                            axis=1),
    #                                  n=len(OFFSETS))
    # np.save('val_pred_lstm.npy', val_preds)
    # np.save('val_ext_lstm.npy', extend_preds)
    # np.save('val_gt_lstm.npy', gt)
