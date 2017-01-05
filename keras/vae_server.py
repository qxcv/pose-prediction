#!/usr/bin/env python3

"""HTTP API for motion-generating VAE. Ridiculous hack to allow myself to
evaluate Keras models from Matlab, without futzing around with virtualenvs from
inside Matlab's Python bindings"""

from common import insert_junk_entries
from flask import Flask, request, json
from keras.models import load_model
import numpy as np
from threading import local

MODEL_PATH = './seq-vae-long.bak/models/epoch-100-dec.h5'
STD_MEAN_PATH = './seq-vae-long/meta/std_mean.json'
app = Flask(__name__)
_thread_cache = local()


def get_model():
    # Use a thread-local in case this is run in a multi-threaded setup
    if not hasattr(_thread_cache, 'model'):
        _thread_cache.model = load_model(MODEL_PATH)
    return _thread_cache.model


def get_std_mean():
    if not hasattr(_thread_cache, 'std_mean'):
        with open(STD_MEAN_PATH, 'r') as fp:
            to_load = json.load(fp)
            std = np.array(to_load['std'], dtype=np.number)
            mean = np.array(to_load['mean'], dtype=np.number)
            _thread_cache.std_mean = (std, mean)
    return _thread_cache.std_mean


@app.route("/pose-sequence")
def generate_sequence():
    json_noise = json.loads(request.values['noise'])
    # Force array to be numeric (as opposed to, say, a list of strings, or a
    # dictionary, or some nesting of arbitrary JSON structures)
    noise = np.array(json_noise, dtype=np.number)

    std, mean = get_std_mean()

    model = get_model()
    result = model.predict(noise)
    result = result * std + mean
    result = insert_junk_entries(result)

    return json.jsonify(result.tolist())


if __name__ == '__main__':
    get_model()
    app.run('localhost', 5053)
