#!/usr/bin/env python3
"""Generate a sequence of poses using a variational autoencoder. There's
actually only one latent vector for the whole sequence; I don't expect this
will work well, since noise usually needs to be of the same shape as real data
(e.g. DCGAN paper recommends image-shaped noise)."""

from keras.models import Model
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input, Lambda, RepeatVector
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
import keras.backend as K
import numpy as np
from os import path, makedirs
from scipy.io import savemat
import json

from common import insert_junk_entries
from generate_seq_gan import load_data

np.random.seed(2372143511)

WORK_DIR = './seq-vae'
MODEL_DIR = path.join(WORK_DIR, 'models')
LOG_DIR = path.join(WORK_DIR, 'logs')
POSE_OUT_DIR = path.join(WORK_DIR, 'poses')
META_DIR = path.join(WORK_DIR, 'meta')
# Planning to start with modest prediction lengths and then work up from there.
SEQ_LENGTH = 8
NOISE_DIM = 100
BATCH_SIZE = 64
SEQ_SKIP = 3
POSES_TO_SAVE = 32
INIT_LR = 0.0001


def make_decoder(pose_size):
    x = in_layer = Input(shape=(NOISE_DIM,))
    x = Dense(128)(x)
    x = Activation('relu')(x)
    # Leaving BN out for now. Not sure how to make it work with LSTMs :/
    # x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = RepeatVector(SEQ_LENGTH)(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = out_layer = TimeDistributed(Dense(pose_size))(x)

    decoder = Model(input=[in_layer], output=[out_layer])

    return decoder


def make_encoder(pose_size):
    x = in_layer = Input(shape=(SEQ_LENGTH, pose_size,))
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = LSTM(1000, return_sequences=False)(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    mean = Dense(NOISE_DIM)(x)
    log_std = Dense(NOISE_DIM)(x)
    std = Lambda(lambda l: K.exp(l))(log_std)

    encoder = Model(input=[in_layer], output=[mean, std, log_std])

    return encoder


def make_vae(pose_size):
    encoder = make_encoder(pose_size)
    decoder = make_decoder(pose_size)

    # Sample from encoder's output distribution
    encoder_in = Input(shape=(SEQ_LENGTH, pose_size,))
    mean, std, log_std = encoder(encoder_in)
    def make_noise(layers):  # noqa
        mean, std = layers
        noise = K.random_normal(shape=K.shape(std), mean=0., std=1.)
        return noise * std + mean
    latent = Lambda(make_noise)([mean, std])

    # Run latent variables through decoder
    decoder_out = decoder(latent)

    vae = Model(input=[encoder_in], output=[decoder_out])

    # Mean loss is \|mu\|_2^2, std loss is tr(sig) - log(det(sig))
    kl_loss = K.sum(K.square(mean) + std - log_std, axis=-1) - NOISE_DIM

    def kl_div(true, pred):
        return K.mean(kl_loss)

    def likelihood(true, pred):
        diff = true - pred
        return K.mean(K.sum(K.sum(K.square(diff), axis=-1), axis=-1))

    def loss(x, x_hat):
        # Log likelihood loss is just squared L2
        diff = x - x_hat
        l2_loss = K.sum(K.sum(K.square(diff), axis=-1), axis=-1)

        return l2_loss + kl_loss

    vae_opt = RMSprop(lr=INIT_LR, clipnorm=1.0)
    vae.compile(vae_opt, loss=loss, metrics=[kl_div, likelihood])

    return vae, encoder, decoder


def train_model(train_X, val_X, mean, std):
    assert train_X.ndim == 3, train_X.ndim
    total_X, time_steps, out_shape = train_X.shape
    vae, encoder, decoder = make_vae(out_shape)

    try:
        makedirs(POSE_OUT_DIR)
    except FileExistsError:
        pass
    try:
        makedirs(MODEL_DIR)
    except FileExistsError:
        pass

    def sample_trajectories(epoch, logs={}):
        gen_poses = decoder.predict(np.random.randn(POSES_TO_SAVE, NOISE_DIM))
        gen_poses = gen_poses * std + mean
        gen_poses = insert_junk_entries(gen_poses)

        train_inds = np.random.permutation(len(train_X))[:POSES_TO_SAVE]
        train_poses = train_X[train_inds] * std + mean
        train_poses = insert_junk_entries(train_poses)

        val_inds = np.random.permutation(len(val_X))[:POSES_TO_SAVE]
        val_poses = val_X[val_inds] * std + mean
        val_poses = insert_junk_entries(val_poses)

        out_path = path.join(POSE_OUT_DIR, 'preds-epoch-%d.mat' % (epoch + 1))
        print('\nSaving samples to', out_path)
        savemat(out_path, {
            'gen_poses': gen_poses,
            'train_poses': train_poses,
            'val_poses': val_poses
        })

    def save_encoder_decoder(epoch, logs={}):
        model_path = path.join(MODEL_DIR, 'epoch-{epoch:02d}'.format(
            epoch=epoch
        ))
        encoder_path = model_path + '-enc.h5'
        decoder_path = model_path + '-dec.h5'
        print('Saving encoder to %s' % encoder_path)
        encoder.save(encoder_path)
        print('Saving decoder to %s' % decoder_path)
        decoder.save(decoder_path)

    print('Training recurrent VAE')
    cb_list = [
        LambdaCallback(on_epoch_end=sample_trajectories),
        LambdaCallback(on_epoch_end=save_encoder_decoder),
        ReduceLROnPlateau(patience=10)
    ]
    # We reverse input data to make encoder LSTM's job easier
    vae.fit(train_X[:, ::-1, :], train_X, validation_data=(val_X[:, ::-1, :], val_X),
            shuffle=True, batch_size=BATCH_SIZE, nb_epoch=1000,
            callbacks=cb_list)

    return vae, encoder, decoder


if __name__ == '__main__':
    print('Loading data')
    train_X, val_X, mean, std = load_data(SEQ_LENGTH, SEQ_SKIP)
    print('Data loaded')

    std_mean_path = path.join(META_DIR, 'std_mean.json')
    print('Saving mean/std to %s' % std_mean_path)
    try:
        makedirs(META_DIR)
    except FileExistsError:
        pass
    with open(std_mean_path, 'w') as fp:
        to_dump = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        json.dump(to_dump, fp)

    model = train_model(train_X, val_X, mean, std)
