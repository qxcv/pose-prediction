#!/usr/bin/env python3
"""Generate a sequence of poses using a variational autoencoder. There's
actually only one latent vector for the whole sequence; I don't expect this
will work well, since noise usually needs to be of the same shape as real data
(e.g. DCGAN paper recommends image-shaped noise)."""

from argparse import ArgumentParser
import json
from os import path, makedirs
import sys

from keras.models import Model, load_model
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input, Lambda, RepeatVector
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
import keras.backend as K

import numpy as np

from scipy.io import savemat

from common import insert_junk_entries
from generate_seq_gan import load_data

np.random.seed(2372143511)


def make_decoder(pose_size, seq_length, noise_dim):
    x = in_layer = Input(shape=(noise_dim,))
    x = Dense(128)(x)
    x = Activation('relu')(x)
    # Leaving BN out for now. Not sure how to make it work with LSTMs :/
    # x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = RepeatVector(seq_length)(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = out_layer = TimeDistributed(Dense(pose_size))(x)

    decoder = Model(input=[in_layer], output=[out_layer])

    return decoder


def make_encoder(pose_size, seq_length, noise_dim):
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
    x = Dense(500)(x)
    x = last_shared = Activation('relu')(x)
    mean = Dense(noise_dim)(last_shared)
    var_in = Dense(noise_dim)(last_shared)
    var = Activation('softplus')(var_in)

    encoder = Model(input=[in_layer], output=[mean, var])

    return encoder


def copy_weights(source, dest):
    assert len(source.layers) == len(dest.layers)
    for src_layer, dest_layer in zip(source.layers, dest.layers):
        assert dest_layer.__class__ == src_layer.__class__
        dest_layer.set_weights(src_layer.get_weights())


def make_vae(pose_size, args):
    seq_in_length = args.seq_in_length
    seq_out_length = args.seq_out_length
    noise_dim = args.noise_dim
    init_lr = args.init_lr

    encoder = make_encoder(pose_size, seq_in_length, noise_dim)
    if args.encoder_path:
        print('Reloading encoder weights from %s' % args.encoder_path)
        saved_encoder = load_model(args.encoder_path)
        copy_weights(saved_encoder, encoder)

    decoder = make_decoder(pose_size, seq_out_length, noise_dim)
    if args.decoder_path:
        print('Reloading decoder weights from %s' % args.decoder_path)
        saved_decoder = load_model(args.decoder_path)
        copy_weights(saved_decoder, decoder)

    # Sample from encoder's output distribution
    encoder_in = Input(shape=(seq_in_length, pose_size,))
    mean, var = encoder(encoder_in)
    log_std = Lambda(lambda var: 0.5 * K.log(var))(var)
    std = Lambda(lambda var: K.sqrt(var))(var)
    def make_noise(layers):  # noqa
        mean, std = layers
        noise = K.random_normal(shape=K.shape(std), mean=0., std=1.)
        return noise * std + mean
    latent = Lambda(make_noise)([mean, std])

    # Run latent variables through decoder
    decoder_out = decoder(latent)

    vae = Model(input=[encoder_in], output=[decoder_out])

    def kl_inner(l):
        # Mean loss is \|mu\|_2^2, std loss is tr(sig) - log(det(sig))
        mean, std, log_std = l
        return K.sum(K.square(mean) + std - log_std, axis=1) - noise_dim

    kl_loss = Lambda(kl_inner)([mean, std, log_std])

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

    vae_opt = RMSprop(lr=init_lr, clipnorm=1.0)
    vae.compile(vae_opt, loss=loss, metrics=[kl_div, likelihood])

    return vae, encoder, decoder


def train_model(train_X, val_X, mean, std, args):
    assert train_X.ndim == 3, train_X.ndim
    total_X, time_steps, out_shape = train_X.shape
    vae, encoder, decoder = make_vae(out_shape, args)

    try:
        makedirs(args.pose_dir)
    except FileExistsError:
        pass
    try:
        makedirs(args.model_dir)
    except FileExistsError:
        pass

    def sample_trajectories(epoch, logs={}):
        poses_to_save = args.poses_to_save
        gen_poses = decoder.predict(np.random.randn(poses_to_save, args.noise_dim))
        gen_poses = gen_poses * std + mean
        gen_poses = insert_junk_entries(gen_poses)

        train_inds = np.random.permutation(len(train_X))[:poses_to_save]
        train_poses = train_X[train_inds] * std + mean
        train_poses = insert_junk_entries(train_poses)

        val_inds = np.random.permutation(len(val_X))[:poses_to_save]
        val_poses = val_X[val_inds] * std + mean
        val_poses = insert_junk_entries(val_poses)

        out_path = path.join(args.pose_dir, 'preds-epoch-%d.mat' % (epoch + 1))
        print('\nSaving samples to', out_path)
        savemat(out_path, {
            'gen_poses': gen_poses,
            'train_poses': train_poses,
            'val_poses': val_poses
        })

    def model_paths(epoch, logs={}):
        model_path = path.join(args.model_dir, 'epoch-{epoch:02d}'.format(
            epoch=epoch
        ))
        encoder_path = model_path + '-enc.h5'
        decoder_path = model_path + '-dec.h5'
        return encoder_path, decoder_path

    def save_encoder_decoder(epoch, logs={}):
        encoder_path, decoder_path = model_paths(epoch, logs)
        print('Saving encoder to %s' % encoder_path)
        encoder.save(encoder_path)
        print('Saving decoder to %s' % decoder_path)
        decoder.save(decoder_path)

    def save_state(epoch, logs={}):
        # Save all paths and arguments to file. Good for resumption of
        # training.
        encoder_path, decoder_path = model_paths(epoch, logs)
        extra_args = sys.argv[1:]
        config_dest = args.config_path
        data = {
            'encoder_path': encoder_path,
            'decoder_path': decoder_path,
            'args': extra_args
        }
        print('Saving config to', config_dest)
        with open(config_dest, 'w') as fp:
            json.dump(data, fp, indent=2)

    def check_prediction_accuracy(epoch, logs={}):
        # TODO: Record L2 distance at each time offset (esp. vs 'extend'
        # baseline)
        pass

    print('Training recurrent VAE')
    cb_list = [
        LambdaCallback(on_epoch_end=sample_trajectories),
        LambdaCallback(on_epoch_end=save_encoder_decoder),
        LambdaCallback(on_epoch_end=save_state),
        LambdaCallback(on_epoch_end=check_prediction_accuracy),
        ReduceLROnPlateau(patience=10)
    ]
    # We reverse input data to make encoder LSTM's job easier
    vae.fit(train_X[:, ::-1, :], train_X, validation_data=(val_X[:, ::-1, :], val_X),
            shuffle=True, batch_size=args.batch_size, nb_epoch=1000,
            callbacks=cb_list)

    return vae, encoder, decoder


# TODO: Things which are probably beneficial to change over time:
# - Sequence length (both input and output)
# - Diversity of samples (in terms of # of distinct action classes). Need to
#   clean data first, though.
# - Coefficient on KL divergence term
# - Learning rate (already changed via plateau thing)

parser = ArgumentParser()
parser.add_argument('--lr', type=float, dest='init_lr', default=0.0001,
                    help='initial learning rate')
parser.add_argument('--work-dir', type=str, dest='work_dir', default='./seq-vae',
                    help='parent directory to store state')
parser.add_argument('--seq-in-length', type=int, dest='seq_in_length', default=8,
                    help='length of input sequence')
parser.add_argument('--seq-out-length', type=int, dest='seq_out_length', default=8,
                    help='length of sequence to predict (may overlap)')
parser.add_argument('--save-poses', type=int, dest='poses_to_save', default=32,
                    help='number of sample poses to save at each epoch')
parser.add_argument('--seq-skip', type=int, dest='seq_skip', default=3,
                    help='factor by which to temporally downsample data')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=64,
                    help='size of training batch')
parser.add_argument('--noise-dim', type=int, dest='noise_dim', default=64,
                    help='number of latent variables')
parser.add_argument('--no-resume', action='store_false', dest='resume', default=True,
                    help='stop automatic training resumption from checkpoint')


def add_extra_paths(args):
    wd = args.work_dir
    args.meta_dir = path.join(wd, 'meta')
    args.log_dir = path.join(wd, 'logs')
    args.pose_dir = path.join(wd, 'poses')
    args.model_dir = path.join(wd, 'models')
    args.config_path = path.join(wd, 'config.json')


def load_args():
    args = parser.parse_args()

    # Extra paths
    add_extra_paths(args)

    # Have to re-parse args
    if args.resume and path.exists(args.config_path):
        print('Reloading config from', args.config_path)
        with open(args.config_path, 'r') as fp:
            config = json.load(fp)
        extra_args = config['args']
        arg_list = extra_args + sys.argv[1:]
        args = parser.parse_args(arg_list)
        add_extra_paths(args)
        args.encoder_path = config['encoder_path']
        args.decoder_path = config['decoder_path']
    else:
        args.encoder_path = args.decoder_path = None

    return args


if __name__ == '__main__':
    args = load_args()

    print('Loading data')
    seq_length = max(args.seq_in_length, args.seq_out_length)
    train_X, val_X, mean, std = load_data(seq_length, args.seq_skip)
    print('Data loaded')

    std_mean_path = path.join(args.meta_dir, 'std_mean.json')
    print('Saving mean/std to %s' % std_mean_path)
    try:
        makedirs(args.meta_dir)
    except FileExistsError:
        pass
    with open(std_mean_path, 'w') as fp:
        to_dump = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        json.dump(to_dump, fp)

    model = train_model(train_X, val_X, mean, std, args)
