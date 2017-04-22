#!/usr/bin/env python3
"""LSTM-3LR model from ERD paper"""

from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, TimeDistributed, Masking
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

from common import huber_loss, VariableGaussianNoise, GaussianRamper, \
    CUSTOM_OBJECTS, load_mocap_data, scrape_sequences, insert_junk_entries, \
    NOISE_SCHEDULE, get_offset_losses

np.random.seed(2372143511)

SEQ_LENGTH = 129
WEIGHTS_PATH = './best-lstm-3lr-weights.h5'


def make_model_train(shape, mask_value=0.0):
    step_data_size = shape[-1]

    x = in_layer = Input(shape=shape)
    x = Masking(mask_value=mask_value)(x)
    x = VariableGaussianNoise(sigma=0.01)(x)
    x = TimeDistributed(Dense(500))(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = LSTM(1000, return_sequences=True)(x)
    out_layer = TimeDistributed(Dense(step_data_size))(x)

    model = Model(inputs=[in_layer], outputs=[out_layer])

    return model


def make_model_predict(train_model, mask_value=0.0):
    assert len(train_model.input_shape) == 3

    step_data_size = train_model.input_shape[2]

    x = in_layer = Input(batch_shape=(1, 1, step_data_size))
    x = Masking(mask_value=mask_value)(x)
    x = VariableGaussianNoise(sigma=1e-10)(x)
    x = TimeDistributed(Dense(500))(x)
    x = LSTM(1000, return_sequences=True, stateful=True)(x)
    x = LSTM(1000, return_sequences=True, stateful=True)(x)
    x = LSTM(1000, return_sequences=True, stateful=True)(x)
    out_layer = TimeDistributed(Dense(step_data_size))(x)

    pred_model = Model(inputs=[in_layer], outputs=[out_layer])

    pred_model.compile(optimizer='rmsprop', loss=huber_loss)

    assert len(pred_model.layers) == len(train_model.layers)

    for new_layer, orig_layer in zip(pred_model.layers, train_model.layers):
        new_layer.set_weights(orig_layer.get_weights())

    return pred_model


def train_model(train_X,
                train_Y,
                val_X,
                val_Y,
                mask_value=0.0,
                save_path=WEIGHTS_PATH):
    assert train_X.ndim == 3

    model = make_model_train(train_X.shape[1:], mask_value=mask_value)

    # objective = huber_loss
    objective = 'mse'
    model.compile(optimizer='rmsprop', loss=objective, metrics=[objective])

    print('Fitting to data')
    mod_check = ModelCheckpoint(save_path, save_best_only=True)
    estop = EarlyStopping(min_delta=0, patience=100)
    ramper = GaussianRamper(patience=10, schedule=NOISE_SCHEDULE)
    callbacks = [mod_check, estop, ramper]
    model.fit(
        train_X,
        train_Y,
        batch_size=8,
        validation_data=(val_X, val_Y),
        epochs=2000,
        shuffle=True,
        callbacks=callbacks)

    return model


if __name__ == '__main__':
    print('Loading data')
    (train_X, train_Y, train_means, train_stds, val_X, val_Y, val_means,
     val_stds) = load_mocap_data(SEQ_LENGTH)
    print('Data loaded')

    model = None
    try:
        print('Loading model')
        model = load_model(WEIGHTS_PATH, CUSTOM_OBJECTS)
    except OSError:
        print('Load failed, building model anew')
    if model is None:
        model = train_model(train_X, train_Y, val_X, val_Y)

    print('Computing l2 losses from sampled poses')
    pred_model = make_model_predict(model)
    print('{:>12} {:>12}'.format('+t (ms)', 'l2 loss'))
    for offset, mean_loss in get_offset_losses(pred_model, val_X, val_Y,
                                               val_means, val_stds):
        print('{:>12} {:>12}'.format(offset * 20, mean_loss))

    print('Scraping predictions for visualisation')
    results = scrape_sequences(pred_model, val_X, val_means, val_stds, 1,
                               SEQ_LENGTH)
    to_write = insert_junk_entries(results[0])
    np.savetxt('prediction_3lr.txt', to_write, delimiter=',')
