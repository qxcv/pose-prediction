#!/usr/bin/env python3

"""VAE which generates a single pose at a time. Might end up using this as the
basis for a full-on sequence encoder/decoder."""

from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Input, Lambda
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
import numpy as np
import re
from glob import glob
from os import path, makedirs
from scipy.io import savemat
from multiprocessing import Pool

from common import GOOD_MOCAP_INDS, insert_junk_entries

np.random.seed(2372143511)

POSE_OUT_DIR = 'vae/poses'
MODEL_DIR = 'vae/models'
NOISE_DIM = 25
BATCH_SIZE = 64
INIT_LR = 0.001
POSES_TO_SAVE = 256


def make_decoder(pose_size):
    x = in_layer = Input(shape=(NOISE_DIM,))
    x = Dense(500, input_dim=NOISE_DIM)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = out_layer = Dense(pose_size)(x)

    decoder = Model(input=[in_layer], output=[out_layer])

    return decoder


def make_encoder(pose_size):
    x = in_layer = Input(shape=(pose_size,))
    x = Dense(500, input_shape=(pose_size,))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # Need to output both mean and (diagonal) standard deviation
    mean = Dense(NOISE_DIM)(x)
    log_std = Dense(NOISE_DIM)(x)
    std = Lambda(lambda l: K.exp(l))(log_std)

    encoder = Model(input=[in_layer], output=[mean, std, log_std])

    return encoder


def make_vae(pose_size):
    """Make a VAE. See ``variational_autoencoder.py`` in Keras examples for
    good illustration of how to do this The Right Way(tm)."""
    encoder = make_encoder(pose_size)
    decoder = make_decoder(pose_size)

    # Sample from encoder's output distribution
    encoder_in = Input(shape=(pose_size,))
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
    # XXX: Should I be taking the mean or something with KL divergence?
    # Having a loss that scales linearly in the number of dimensions is
    # bizarre.
    kl_loss = K.sum(K.square(mean) + std - log_std, axis=-1)

    # I've split the loss into three functions. kl_div and likelihood are
    # intended as metrics, so that I can monitor individual loss quantities.
    # loss() is the actual model objective, and is the sum of those two
    # constituents.

    def kl_div(true, pred):
        return K.mean(kl_loss)

    def likelihood(true, pred):
        diff = true - pred
        return K.mean(K.sum(K.square(diff), axis=-1))

    def loss(x, x_hat):
        # Log likelihood loss is just squared L2
        diff = x - x_hat
        l2_loss = K.sum(K.square(diff), axis=-1)

        return l2_loss + kl_loss

    # Compile VAE here so that we can get losses correct
    vae_opt = RMSprop(lr=INIT_LR)
    vae.compile(vae_opt, loss=loss, metrics=[kl_div, likelihood])

    # Shouldn't have to compile the encoder and decoder separately, AFAICT.
    # Just compile the VAE and all will be well.
    return vae, encoder, decoder


def train_model(train_X, val_X, mean, std):
    # TODO: Incorporate standardisation (like original GAN file did)
    assert train_X.ndim == 2, train_X.ndim
    total_X, out_shape = train_X.shape
    vae, encoder, decoder = make_vae(train_X.shape[1])

    # Predictions will be put in here
    try:
        makedirs(POSE_OUT_DIR)
    except FileExistsError:
        pass

    # Model checkpoints will be put here
    try:
        makedirs(MODEL_DIR)
    except FileExistsError:
        pass

    def sample_poses(epoch, logs={}):
        """Save some poses for comparison."""
        gen_poses = decoder.predict(np.random.randn(POSES_TO_SAVE, NOISE_DIM))
        # gen_poses = gen_poses * std + mean
        gen_poses = insert_junk_entries(gen_poses)

        train_inds = np.random.permutation(len(train_X))[:POSES_TO_SAVE]
        train_poses = train_X[train_inds]
        train_poses = insert_junk_entries(train_poses)

        val_inds = np.random.permutation(len(val_X))[:POSES_TO_SAVE]
        val_poses = val_X[val_inds]
        val_poses = insert_junk_entries(val_poses)

        out_path = path.join(POSE_OUT_DIR, 'preds-epoch-%d.mat' % (epoch + 1))
        print('\nSaving samples to', out_path)
        savemat(out_path, {
            'gen_poses': gen_poses,
            'train_poses': train_poses,
            'val_poses': val_poses
        })

    print('Training VAE')
    model_path = path.join(MODEL_DIR, 'weights-{epoch:02d}-{val_loss:.2f}.h5')
    cb_list = [
        LambdaCallback(on_epoch_end=sample_poses),
        ModelCheckpoint(model_path),
        ReduceLROnPlateau(patience=5)
    ]
    vae.fit(train_X, train_X, validation_data=(val_X, val_X),
            shuffle=True, batch_size=BATCH_SIZE, nb_epoch=1000,
            callbacks=cb_list)

    return vae, encoder, decoder


def prepare_data_file(filename):
    poses = np.loadtxt(filename, delimiter=',')
    assert poses.ndim == 2 and poses.shape[1] == 99, poses.shape

    # Take out zero features
    zero_inds, = np.nonzero((poses != 0).any(axis=0))
    assert (zero_inds == GOOD_MOCAP_INDS).all(), zero_inds
    poses = poses[:, GOOD_MOCAP_INDS]

    return poses


def is_valid(data):
    return np.isfinite(data).all()


_fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')
def _mapper(filename):  # noqa
    base = path.basename(filename)
    meta = _fnre.match(base).groupdict()
    subj_id = int(meta['subject'])

    return prepare_data_file(filename), subj_id


def load_data():

    filenames = glob('h36m-3d-poses/expmap_*.txt.gz')

    train_X_blocks = []
    test_X_blocks = []

    with Pool() as pool:
        for X, subj_id in pool.map(_mapper, filenames):
            if subj_id == 5:
                # subject 5 is for testing
                test_X_blocks.append(X)
            else:
                train_X_blocks.append(X)

    train_X = np.concatenate(train_X_blocks, axis=0)
    del train_X_blocks
    test_X = np.concatenate(test_X_blocks, axis=0)
    del test_X_blocks

    mean = train_X.mean(axis=0).reshape((1, -1))
    std = train_X.std(axis=0).reshape((1, -1))

    # train_X = (train_X - mean) / std
    # test_X = (test_X - mean) / std

    assert is_valid(train_X)
    assert is_valid(test_X)
    assert is_valid(mean)
    assert is_valid(std)

    return train_X, test_X,  mean, std


if __name__ == '__main__':
    print('Loading data')
    train_X, val_X, mean, std = load_data()
    print('Data loaded')

    model = train_model(train_X, val_X, mean, std)
