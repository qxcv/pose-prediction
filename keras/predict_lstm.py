#!/usr/bin/env python3

"""LSTM-based model for predicting poses, one frame at a time"""

from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, LSTM, TimeDistributed, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import numpy as np

from common import huber_loss, convert_2d_seq, unmap_predictions, pck, THRESHOLDS

np.random.seed(2372143511)

# Will result in learning mappings from sequences of SEQ_LENGTH - 1 to
# SEQ_LENGTH - 1
SEQ_LENGTH = 129


def make_model_train(shape):
    # Shape is (t, d), so step_data_size is size of data passed to the LSTM at
    # each time step.
    step_data_size = shape[-1]

    x = in_layer = Input(shape=shape)
    x = LSTM(128, return_sequences=True)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    out_layer = TimeDistributed(Dense(step_data_size))(x)

    model = Model(input=[in_layer], output=[out_layer])

    return model


def make_model_predict(train_model):
    # Training on a stateless model is easy, but we need a stateful model to
    # make real predictions. Hence, we have to do this stupid Caffe-style
    # nonsense.
    assert len(train_model.input_shape) == 3

    step_data_size = train_model.input_shape[2]

    x = in_layer = Input(batch_shape=(1, 1, step_data_size))
    x = LSTM(128, return_sequences=True, stateful=True)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    out_layer = TimeDistributed(Dense(step_data_size))(x)

    pred_model = Model(input=[in_layer], output=[out_layer])

    pred_model.compile(optimizer='rmsprop', loss=huber_loss)

    assert len(pred_model.layers) == len(train_model.layers)

    for new_layer, orig_layer in zip(pred_model.layers, train_model.layers):
        new_layer.set_weights(orig_layer.get_weights())

    return pred_model


def train_model(train_X, train_Y, val_X, val_Y):
    assert train_X.ndim == 3
    step_data_size = train_X.shape[-1]

    model = make_model_train(train_X.shape[1:])

    objective = huber_loss
    model.compile(optimizer='rmsprop', loss=objective, metrics=[objective])

    print('Fitting to data')
    mod_check = ModelCheckpoint('./best-lstm-weights.h5', save_best_only=True)
    estop = EarlyStopping(min_delta=0, patience=250)
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


def scrape_sequences(model, data, num_to_scrape):
    """Run the LSTM model over some data one step at a time for
    (SEQ_LENGTH-1)/2 samples, then try to predict over the rest of the
    sequence. Do that num_to_scrape times, choosing samples randomly."""
    sel_indices = np.random.choice(np.arange(data.shape[0]),
                                   size=num_to_scrape,
                                   replace=True)

    assert data.ndim == 3
    assert data.shape[1] == SEQ_LENGTH

    train_k = data.shape[1] // 2
    all_preds = np.zeros((num_to_scrape,) + data.shape[1:])

    for pi, ind in enumerate(sel_indices):
        seq = data[ind]
        model.reset_states()
        preds = np.zeros(data.shape[1:])

        # Feed on GT
        for i in range(train_k):
            preds[i, :] = model.predict(seq[i].reshape((1, 1, -1)))

        # Feed on own preds
        for i in range(train_k, data.shape[1]):
            preds[i, :] = model.predict(preds[i-1, :].reshape((1, 1, -1)))

        all_preds[pi, :, :] = preds

    return all_preds


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

    print('Checking prediction accuracy')
    n = val_X.shape[1] - 1
    gt = unmap_predictions(val_Y[:, 1:], n)
    val_preds = unmap_predictions(model.predict(val_X)[:, :-1], n)
    extend_preds = unmap_predictions(val_X[:, :-1], n)
    np.save('val_pred_lstm.npy', val_preds)
    np.save('val_ext_lstm.npy', extend_preds)
    np.save('val_gt_lstm.npy', gt)

    pred_pcks = pck(gt, val_preds, THRESHOLDS, np.arange(n))
    ext_pcks = pck(gt, extend_preds, THRESHOLDS, np.arange(n))
    assert pred_pcks.keys() == ext_pcks.keys()
    print('metric\t\t\tpred\t\t\text')
    for k in pred_pcks.keys():
        print('{}\t\t{:g}\t\t{:g}'.format(k, pred_pcks[k], ext_pcks[k]))

    print('Scraping predictions')
    pred_model = make_model_predict(model)
    mapped_results = scrape_sequences(pred_model, val_X, 100)
    results = unmap_predictions(mapped_results, n=val_X.shape[1])
    np.save('lstm_scrapes.npy', results)
