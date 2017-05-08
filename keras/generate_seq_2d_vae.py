#!/usr/bin/env python3
"""Try to predict only future pose in 2D. Will have to merge with
generate_seq_act_vae.py."""

from argparse import ArgumentParser
import json
from os import path, makedirs
import sys

from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input, Lambda, RepeatVector
from keras.models import Model, load_model
from keras.optimizers import RMSprop

import numpy as np

import h5py

from common import VariableScaler, ScaleRamper
from p2d_loader import preprocess_sequence, reconstruct_poses

VAL_FRAC = 0.2

np.random.seed(2372143511)


def make_decoder(pose_size, seq_length, noise_dim):
    x = in_layer = Input(shape=(noise_dim, ), name='dec_in')
    x = Dense(128)(x)
    x = Activation('relu')(x)
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
    x = in_layer = Input(shape=(seq_length, pose_size, ), name='enc_in')
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
    mean = Dense(noise_dim, name='enc_mean')(last_shared)
    var_in = Dense(noise_dim)(last_shared)
    var = Activation('softplus', name='enc_var')(var_in)

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
    encoder_in = Input(shape=(seq_in_length, pose_size, ))
    mean, var = encoder(encoder_in)
    log_std = Lambda(
        lambda var: 0.5 * K.log(var),
        output_shape=(noise_dim, ),
        name='log_std')(var)
    std = Lambda(
        lambda var: K.sqrt(var), output_shape=(noise_dim, ), name='std')(var)

    def make_noise(layers):  # noqa
        mean, std = layers
        noise = K.random_normal(shape=K.shape(std), mean=0., stddev=1.)
        return noise * std + mean

    latent = Lambda(
        make_noise, name='make_noise', output_shape=(noise_dim, ))([mean, std])

    # Run latent variables through decoder
    decoder_out = decoder(latent)

    def kl_inner(l):
        # Mean loss is \|mu\|_2^2, std loss is tr(sig) - log(det(sig))
        mean, std, log_std = l
        return K.sum(K.square(mean) + std - log_std, axis=1) - noise_dim

    kl_loss = Lambda(
        kl_inner, output_shape=(None, ), name='kl_loss')([mean, std, log_std])
    kl_loss_scale = VariableScaler(1.0, name='kl_scale')(kl_loss)
    # XXX: This is a silly hack to make sure that kl_loss_scale actually
    # appears in the layer list (thereby allowing the scale ramper to access
    # it). Should figure out a more principled way of achieving what I want
    # (changing the balance between several objectives over time).
    decoder_out = Lambda(lambda a: a[0])([decoder_out, kl_loss_scale])

    def likelihood(true, pred):
        diff = true - pred
        return K.mean(K.sum(K.sum(K.square(diff), axis=-1), axis=-1))

    def kl_div(true, pred):
        return K.mean(kl_loss_scale)

    def loss(y_true, y_pred):
        diff = y_true - y_pred
        l2_loss = K.sum(K.sum(K.square(diff), axis=-1), axis=-1)

        return l2_loss + kl_loss_scale

    vae = Model(input=[encoder_in], output=[decoder_out])
    vae_opt = RMSprop(lr=init_lr, clipnorm=1.0)
    vae.compile(vae_opt, loss=loss, metrics=[kl_div, likelihood])

    return vae, encoder, decoder


def load_data(data_file, seq_in_length, seq_out_length, seq_skip):
    total_seq_len = max(seq_in_length, seq_out_length)

    train_X_blocks = []
    val_X_blocks = []
    train_Y_blocks = []
    val_Y_blocks = []

    with h5py.File(data_file, 'r') as fp:
        parents = fp['/parents'].value

        vid_names = list(fp['seqs'])
        val_vid_list = list(vid_names)
        np.random.shuffle(val_vid_list)
        val_count = max(1, int(VAL_FRAC * len(val_vid_list)))
        val_vids = set(val_vid_list[:val_count])

        for vid_name in fp['seqs']:
            poses = fp['/seqs/' + vid_name + '/poses'].value

            assert poses.ndim == 3, poses.shape
            assert len(parents) == poses.shape[2], poses.shape
            assert 2 == poses.shape[1], poses.shape

            relposes = preprocess_sequence(poses, parents)

            for i in range(len(relposes) - seq_skip * total_seq_len + 1):
                full_block = relposes[i:i + seq_skip * total_seq_len:seq_skip]
                # Input is reversed
                in_block = full_block[:seq_in_length][::-1]
                out_block = full_block[len(full_block) - seq_out_length:]

                assert in_block.ndim == 2 \
                    and in_block.shape[0] == seq_in_length, in_block.shape
                assert out_block.ndim == 2 \
                    and out_block.shape[0] == seq_out_length, out_block.shape

                if vid_name in val_vids:
                    train_X_blocks.append(in_block)
                    train_Y_blocks.append(out_block)
                else:
                    val_X_blocks.append(in_block)
                    val_Y_blocks.append(out_block)

    train_X = np.stack(train_X_blocks, axis=0)
    train_Y = np.stack(train_Y_blocks, axis=0)
    val_X = np.stack(val_X_blocks, axis=0)
    val_Y = np.stack(val_Y_blocks, axis=0)

    flat_X = train_X.reshape((-1, train_X.shape[-1]))
    mean = flat_X.mean(axis=0).reshape((1, 1, -1))
    std = flat_X.std(axis=0).reshape((1, 1, -1))
    std[std < 1e-5] = 1
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std
    train_Y = (train_Y - mean) / std
    val_Y = (val_Y - mean) / std
    assert (train_Y[:, :seq_in_length].flatten() == train_X[:, ::-1].flatten()
            ).all()
    assert (
        val_Y[:, :seq_in_length].flatten() == val_X[:, ::-1].flatten()).all()

    return train_X, train_Y, val_X, val_Y, mean, std, parents


def train_model(train_X, train_Y, val_X, val_Y, mean, std, parents, args):
    # Make sure everything is the right shape
    assert val_X.shape[1] == args.seq_in_length, val_X.shape
    assert val_Y.shape[1] == args.seq_out_length, val_Y.shape
    assert train_X.shape[1] == args.seq_in_length, train_X.shape
    assert train_Y.shape[1] == args.seq_out_length, train_Y.shape

    _, _, pose_shape = train_X.shape
    assert train_Y.shape[2] == pose_shape
    vae, encoder, decoder = make_vae(pose_shape, args)

    try:
        makedirs(args.pose_dir)
    except FileExistsError:
        pass
    try:
        makedirs(args.model_dir)
    except FileExistsError:
        pass

    def sample_trajectories(epoch, logs={}):
        epoch += args.extra_epoch
        poses_to_save = args.poses_to_save
        gen_poses = decoder.predict(
            np.random.randn(poses_to_save, args.noise_dim))
        gen_poses = gen_poses * std + mean
        gen_poses = reconstruct_poses(gen_poses, parents)

        train_inds = np.random.permutation(len(train_X))[:poses_to_save]
        train_poses = train_X[train_inds] * std + mean
        train_poses = reconstruct_poses(train_poses, parents)

        val_inds = np.random.permutation(len(val_X))[:poses_to_save]
        val_poses = val_X[val_inds] * std + mean
        val_poses = reconstruct_poses(val_poses, parents)

        out_path = path.join(args.pose_dir, 'preds-epoch-%d.mat' % (epoch + 1))
        print('\nSaving samples to', out_path)
        np.savez(
            out_path,
            poses_gen=gen_poses,
            poses_train=train_poses,
            poses_val=val_poses,
            parents=parents)

    def model_paths(epoch, logs={}):
        model_path = path.join(
            args.model_dir, 'epoch-{epoch:02d}'.format(epoch=epoch))
        encoder_path = model_path + '-enc.h5'
        decoder_path = model_path + '-dec.h5'
        return encoder_path, decoder_path

    def save_encoder_decoder(epoch, logs={}):
        epoch += args.extra_epoch
        encoder_path, decoder_path = model_paths(epoch, logs)
        print('Saving encoder to %s' % encoder_path)
        encoder.save(encoder_path)
        print('Saving decoder to %s' % decoder_path)
        decoder.save(decoder_path)

    def save_state(epoch, logs={}):
        # Save all paths and arguments to file. Good for resumption of
        # training.
        epoch += args.extra_epoch
        encoder_path, decoder_path = model_paths(epoch, logs)
        config_dest = args.config_path
        data = {
            'encoder_path': encoder_path,
            'decoder_path': decoder_path,
            'args': args._all_args,
            'epoch': epoch + 1
        }
        print('Saving config to', config_dest)
        with open(config_dest, 'w') as fp:
            json.dump(data, fp, indent=2)

    def check_prediction_accuracy(epoch, logs={}):
        # Check prediction accuracy over entire validation dataset
        epoch += args.extra_epoch

        print('Calculating prediction accuracies')

        indices = np.random.permutation(len(val_X))[:1000]
        sub_X = val_X[indices]
        sub_Y = val_Y[indices]

        # 'Extend' baseline
        ext_preds = sub_Y[:, args.seq_in_length - 1:args.seq_in_length]
        ext_losses = np.mean(np.sum((sub_Y - ext_preds)**2, axis=2), axis=0)
        del ext_preds

        # VAE baseline.
        K = 5
        vae_losses = np.zeros((sub_Y.shape[0], sub_Y.shape[1], K))
        for i in range(K):
            vae_preds = vae.predict(sub_X)
            vae_losses[:, :, i] = np.sum((sub_Y - vae_preds)**2, axis=2)
        vae_mean_of_K = np.mean(np.mean(vae_losses, axis=2), axis=0)
        vae_best_of_K = np.mean(np.min(vae_losses, axis=2), axis=0)

        # Same as above, but we randomly shuffle inputs before doing
        # prediction. This will tell us whether we're actually learning a sane
        # embedding.
        random_losses = np.zeros((sub_Y.shape[0], sub_Y.shape[1], K))
        fake_X = sub_X[np.random.permutation(len(sub_X))]
        for i in range(K):
            random_preds = vae.predict(fake_X)
            random_losses[:, :, i] = np.sum((sub_Y - random_preds)**2, axis=2)
        random_mean_of_K = np.mean(np.mean(random_losses, axis=2), axis=0)
        random_best_of_K = np.mean(np.min(random_losses, axis=2), axis=0)

        dest_path = path.join(args.acc_dir, 'epoch-%d.npz' % epoch)
        print('Saving accuracies to', dest_path)
        kwargs = {
            'vae_acc_mean_K': vae_mean_of_K,
            'vae_acc_best_K': vae_best_of_K,
            'random_acc_mean_K': random_mean_of_K,
            'random_acc_best_K': random_best_of_K,
            'ext_acc': ext_losses,
            'epoch': epoch,
            'K': K,
        }
        np.savez(dest_path, **kwargs)

    print('Training recurrent VAE')
    cb_list = [
        LambdaCallback(on_epoch_end=sample_trajectories),
        LambdaCallback(on_epoch_end=check_prediction_accuracy),
        LambdaCallback(on_epoch_end=save_encoder_decoder),
        LambdaCallback(on_epoch_end=save_state),
        # patiences of 10 and 6 should ensure that changes are synchronised
        ReduceLROnPlateau(patience=10),
        ScaleRamper(
            patience=6,
            schedule=args.kl_schedule,
            target_name='kl_scale',
            quantity='val_loss')
    ]
    vae.fit(
        train_X,
        train_Y,
        validation_data=(val_X, val_Y),
        shuffle=True,
        batch_size=args.batch_size,
        nb_epoch=1000,
        callbacks=cb_list)

    return vae, encoder, decoder


# TODO: Things which are probably beneficial:
# - Adding action features (from Anoop's home dir)


def some_floats(str_in):
    """Converts comma-seperated float list to Python float tuple"""
    items = str_in.split(',')
    stripped = map(str.strip, items)
    nonempty = filter(bool, stripped)
    floats = map(float, nonempty)
    return tuple(floats)


parser = ArgumentParser()
parser.add_argument(
    '--lr',
    type=float,
    dest='init_lr',
    default=0.0001,
    help='initial learning rate')
parser.add_argument(
    '--work-dir',
    type=str,
    dest='work_dir',
    default='./seq-2d-vae',
    help='parent directory to store state')
parser.add_argument(
    '--seq-in-length',
    type=int,
    dest='seq_in_length',
    default=16,
    help='length of input sequence')
parser.add_argument(
    '--seq-out-length',
    type=int,
    dest='seq_out_length',
    default=16,
    help='length of sequence to predict (may overlap)')
parser.add_argument(
    '--save-poses',
    type=int,
    dest='poses_to_save',
    default=32,
    help='number of sample poses to save at each epoch')
parser.add_argument(
    '--seq-skip',
    type=int,
    dest='seq_skip',
    default=3,
    help='factor by which to temporally downsample data')
parser.add_argument(
    '--batch-size',
    type=int,
    dest='batch_size',
    default=64,
    help='size of training batch')
parser.add_argument(
    '--noise-dim',
    type=int,
    dest='noise_dim',
    default=64,
    help='number of latent variables')
parser.add_argument(
    '--data-file',
    type=str,
    dest='data_file',
    default=None,
    help='HDF5 file containing poses')
# --feat-file is generally used to provide extra context for each frame in the
# encoder. The aim is to help the model learn better latent featuers.
# parser.add_argument('--feat-file', type=str, dest='feat_file', default=None,
#                     help='HDF5 file containing auxiliary, per-frame features')
parser.add_argument(
    '--kl-schedule',
    dest='kl_schedule',
    type=some_floats,
    default=(1.0, ),
    help='KL divergence coefficient (try ramping this up)')
parser.add_argument(
    '--no-resume',
    action='store_false',
    dest='resume',
    default=True,
    help='stop automatic training resumption '
    'from checkpoint')


def add_extra_paths(args):
    wd = args.work_dir
    args.meta_dir = path.join(wd, 'meta')
    args.pose_dir = path.join(wd, 'poses')
    args.model_dir = path.join(wd, 'models')
    args.acc_dir = path.join(wd, 'accs')
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
        args._all_args = arg_list
        args.encoder_path = config['encoder_path']
        args.decoder_path = config['decoder_path']
        args.extra_epoch = config['epoch']
    else:
        args.encoder_path = args.decoder_path = None
        args.extra_epoch = 0
        args._all_args = sys.argv[1:]
        assert args.data_file is not None, "Data file required if not resuming"

    return args


if __name__ == '__main__':
    args = load_args()

    print('Loading data')
    seq_length = max(args.seq_in_length, args.seq_out_length)
    train_X, train_Y, val_X, val_Y, mean, std, parents \
        = load_data(args.data_file, args.seq_in_length, args.seq_out_length,
                    args.seq_skip)
    print('Data loaded')

    print('Making directories')
    for dir_name in ['meta', 'pose', 'model', 'acc']:
        to_make = getattr(args, dir_name + '_dir')
        try:
            makedirs(to_make)
        except FileExistsError:
            pass

    std_mean_path = path.join(args.meta_dir, 'std_mean.json')
    print('Saving mean/std to %s' % std_mean_path)
    with open(std_mean_path, 'w') as fp:
        to_dump = {'mean': mean.tolist(), 'std': std.tolist()}
        json.dump(to_dump, fp)

    model = train_model(train_X, train_Y, val_X, val_Y, mean, std, parents,
                        args)
