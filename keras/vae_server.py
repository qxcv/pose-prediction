#!/usr/bin/env python3

"""HTTP API for motion-generating VAE. Ridiculous hack to allow myself to
evaluate Keras models from Matlab, without futzing around with virtualenvs from
inside Matlab's Python bindings"""

from common import insert_junk_entries
from flask import Flask, request, json
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, LSTM, RepeatVector
import numpy as np
from threading import local

MODEL_PATH = './seq-vae-long.bak/models/epoch-100-dec.h5'
STD_MEAN_PATH = './seq-vae-long/meta/std_mean.json'
# Ugh, silly hardcoded constants for building model :(
NOISE_DIM = 100
POSE_SIZE = 54
app = Flask(__name__)
_thread_cache = local()


def make_stateful():
    x = in_layer = Input(batch_shape=(1, NOISE_DIM,))
    x = Dense(128)(x)
    x = Activation('relu')(x)
    # Leaving BN out for now. Not sure how to make it work with LSTMs :/
    # x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = RepeatVector(1)(x)
    # Doesn't matter if we don't return sequences, since we only need one
    # output anyway. No more TimeDistributed after this
    x = LSTM(500, stateful=True, return_sequences=False)(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = out_layer = Dense(POSE_SIZE)(x)

    return Model(input=[in_layer], output=[out_layer])


def convert_to_stateful(model):
    # Create new model mimicking old one, but with stateful LSTM. Not sure how
    # to do this for general input models.
    stateful_model = make_stateful()
    assert len(model.layers) == len(stateful_model.layers)
    for src_layer, dest_layer in zip(model.layers, stateful_model.layers):
        dest_layer.set_weights(src_layer.get_weights())
    return stateful_model


def get_model():
    # Use a thread-local in case this is run in a multi-threaded setup
    if not hasattr(_thread_cache, 'model'):
        orig_model = load_model(MODEL_PATH)
        _thread_cache.model = convert_to_stateful(orig_model)
    return _thread_cache.model


def get_std_mean():
    if not hasattr(_thread_cache, 'std_mean'):
        with open(STD_MEAN_PATH, 'r') as fp:
            to_load = json.load(fp)
            std = np.array(to_load['std'], dtype=np.number)
            mean = np.array(to_load['mean'], dtype=np.number)
            _thread_cache.std_mean = (std, mean)
    return _thread_cache.std_mean


def predict_seq(model, noise, length):
    assert noise.ndim == 2, noise.shape
    assert noise.shape[0] == 1 and noise.shape[1] == NOISE_DIM, noise.shape
    rv = []
    model.reset_states()
    for i in range(length):
        out = model.predict_on_batch(noise)
        assert out.ndim == 2 and out.shape[0] == 1, out.shape
        rv.append(out.flatten())
    # Not 100% sure that this is the right axis order
    return np.stack(rv)


@app.route("/pose-sequence")
def generate_sequence():
    json_noise = json.loads(request.values['noise'])
    seq_length = int(request.values.get('length', 64))
    # Force array to be numeric (as opposed to, say, a list of strings, or a
    # dictionary, or some nesting of arbitrary JSON structures)
    noise = np.array(json_noise, dtype=np.number)
    if noise.ndim == 1:
        # Use a single row. Matlab's JSON encoder is pretty crap, so it won't
        # let us encode a matrix with a single row (well, actually Matlab's
        # matrix semantics are crap)
        noise = noise.reshape((1, -1))
    # Noise shape is (batch size, noise dimension)
    assert noise.shape[1] == NOISE_DIM, noise.shape

    std, mean = get_std_mean()

    model = get_model()
    result = []
    for row in noise:
        assert row.ndim == 1
        row = row.reshape((1, NOISE_DIM))
        result.append(predict_seq(model, row, seq_length))
    result = np.stack(result)
    result = result * std + mean
    result = insert_junk_entries(result)

    return json.jsonify(result.tolist())


if __name__ == '__main__':
    get_model()
    app.run('localhost', 5053)
