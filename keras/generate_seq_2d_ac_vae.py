#!/usr/bin/env python3
"""Try to predict only future pose in 2D, conditioned on action. The
conditioning strategy is somewhat novel: I use the encoder to generate a PDF
N(mu_x, sigma_x^2), then use the action to generate a second PDF N(mu_a,
sigma_a^2). The resultant latent is sampled from the product of the PDFs (which
is also normal)."""

from argparse import ArgumentParser
import json
from os import path, makedirs
import sys

from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input, Lambda, RepeatVector, Embedding, Dropout
from keras.models import Model, load_model
from keras.optimizers import RMSprop

import numpy as np

from common import VariableScaler, ScaleRamper
from p2d_loader import P2DDataset

VAL_FRAC = 0.2

np.random.seed(2372143511)


def make_decoder(pose_size, seq_length, noise_dim):
    # Idea: might it help to share embedding layers between the three
    # sub-models I have (action encoder, pose encoder, decoder)?

    # XXX: setup is somewhat broken here. Should really be merging the actions
    # in at the LSTM layer, not anywhere else. Same goes for pose encoder
    # attempt below.

    # pose_in_layer = Input(shape=(noise_dim, ), name='dec_pose_in')
    # act_in_layer = Input(shape=(seq_length, pose_size, ), name='dec_act_in')
    # act_embed = Embedding(num_actions, 64, input_length=seq_length)(act_in_layer)
    # x = concatenate([pose_in_layer, act_embed], axis=-1)
    x = pose_in_layer = Input(shape=(noise_dim, ), name='dec_pose_in')
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = RepeatVector(seq_length)(x)
    x = LSTM(1024, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(x)
    x = LSTM(1024, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = out_layer = TimeDistributed(Dense(pose_size))(x)

    # decoder = Model(input=[pose_in_layer, act_in_layer], output=[out_layer])
    decoder = Model(input=[pose_in_layer], output=[out_layer])

    return decoder


def make_pose_encoder(pose_size, seq_length, noise_dim):
    # pose_in_layer = Input(shape=(seq_length, pose_size, ), name='pose_enc_pose_in')
    # act_in_layer = Input(shape=(seq_length, pose_size, ), name='pose_enc_act_in')
    # act_embed = Embedding(num_actions, 64, input_length=seq_length)(act_in_layer)
    # x = concatenate([pose_in_layer, act_embed], axis=-1)
    x = pose_in_layer = Input(shape=(seq_length, pose_size, ), name='pose_enc_pose_in')
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Dense(128))(x)
    x = LSTM(1024, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(x)
    x = LSTM(1024, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(x)
    x = Dense(512)(x)
    x = last_shared = Activation('relu')(x)
    mean = Dense(noise_dim, name='pose_enc_mean')(last_shared)
    var_in = Dense(noise_dim)(last_shared)
    var = Activation('softplus', name='pose_enc_var')(var_in)

    # encoder = Model(input=[pose_in_layer, act_in_layer], output=[mean, var])
    encoder = Model(input=[pose_in_layer], output=[mean, var])

    return encoder


def make_act_encoder(num_actions, seq_length, noise_dim):
    x = in_layer = Input(shape=(seq_length, ), name='act_enc_in')
    x = Embedding(num_actions, 64, input_length=seq_length)(x)
    x = LSTM(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(x)
    mean = Dense(noise_dim, name='act_enc_mean')(x)
    var_in = Dense(noise_dim)(x)
    var = Activation('softplus', name='act_enc_var')(var_in)

    encoder = Model(input=[in_layer], output=[mean, var])

    return encoder


def copy_weights(source, dest):
    assert len(source.layers) == len(dest.layers)
    for src_layer, dest_layer in zip(source.layers, dest.layers):
        assert dest_layer.__class__ == src_layer.__class__
        dest_layer.set_weights(src_layer.get_weights())


def make_vae(pose_size, num_actions, args):
    noise_dim = args.noise_dim
    init_lr = args.init_lr

    pose_encoder = make_pose_encoder(pose_size, seq_length, noise_dim)
    if args.pose_encoder_path:
        print('Reloading pose encoder weights from %s' %
              args.pose_encoder_path)
        saved_pose_encoder = load_model(args.pose_encoder_path)
        copy_weights(saved_pose_encoder, pose_encoder)

    act_encoder = make_act_encoder(num_actions, seq_length, noise_dim)
    if args.act_encoder_path:
        print('Reloading action encoder weights from %s' %
              args.act_encoder_path)
        saved_act_encoder = load_model(args.act_encoder_path)
        copy_weights(saved_act_encoder, act_encoder)

    decoder = make_decoder(pose_size, seq_length, noise_dim)
    if args.decoder_path:
        print('Reloading decoder weights from %s' % args.decoder_path)
        saved_decoder = load_model(args.decoder_path)
        copy_weights(saved_decoder, decoder)

    # Sample from encoder's output distribution
    pose_encoder_in = Input(
        shape=(seq_length, pose_size, ), name='reverse_poses')
    pose_mean, pose_var = pose_encoder(pose_encoder_in)
    # Action encoder only provides a prior
    act_encoder_in = Input(shape=(seq_length, ), name='reverse_actions')
    act_mean, act_var = act_encoder(act_encoder_in)

    pose_std = Lambda(
        lambda var: K.sqrt(var), output_shape=(noise_dim, ),
        name='pose_std')(pose_var)
    act_std = Lambda(
        lambda var: K.sqrt(var), output_shape=(noise_dim, ),
        name='act_std')(act_var)

    def make_noise(layers):  # noqa
        mean, std = layers
        noise = K.random_normal(shape=K.shape(pose_std), mean=0., std=1.)
        return noise * std + mean

    # Only pose-dependent mean/var contribute to the sequence. Action mean/var
    # only serves as (a) regulariser and (b) prior to use during evaluation
    # (trained by KL term).
    latent = Lambda(
        make_noise, name='make_noise',
        output_shape=(noise_dim, ))([pose_mean, pose_std])

    # Run latent variables through decoder
    decoder_out = decoder(latent)

    def kl_inner(l):
        # Mean loss is \|mu\|_2^2, std loss is tr(sig) - log(det(sig))
        mean_post, std_post, mean_pri, std_pri = l
        # In 1D, KL is log(sigma_2) - log(sigma_1)
        # + (sigma_1^2 + (mu_1 - mu_2)^2) / sigma_2^2 - 1/2
        # Need to extend to N dimensions
        noise_dim = K.shape(mean_post)[-1]
        log_post = K.log(std_post)
        log_pri = K.log(std_pri)
        mu_diff = K.square(mean_post - mean_pri)
        var_pri = K.square(std_pri)
        var_post = K.square(std_post)
        inner_rat = (var_post + mu_diff) / var_pri
        return K.sum(log_pri - log_post + inner_rat, axis=-1) - noise_dim / 2.0

    kl_loss = Lambda(
        kl_inner, output_shape=(None, ),
        name='kl_loss')([pose_mean, pose_std, act_mean, act_std])
    kl_loss_scale = VariableScaler(1.0, name='kl_scale')(kl_loss)
    # XXX: This is a silly hack to make sure that kl_loss_scale actually
    # appears in the layer list (thereby allowing the scale ramper to access
    # it). Should figure out a more principled way of achieving what I want
    # (changing the balance between several objectives over time).
    decoder_out = Lambda(
        lambda a: a[0], output_shape=K.int_shape(decoder_out)[1:])(
            [decoder_out, kl_loss_scale])

    def likelihood(true, pred):
        diff = true - pred
        return K.mean(K.sum(K.sum(K.square(diff), axis=-1), axis=-1))

    def kl_div(true, pred):
        return K.mean(kl_loss_scale)

    def loss(y_true, y_pred):
        diff = y_true - y_pred
        l2_loss = K.sum(K.sum(K.square(diff), axis=-1), axis=-1)

        return l2_loss + kl_loss_scale

    vae = Model(input=[pose_encoder_in, act_encoder_in], output=[decoder_out])
    vae_opt = RMSprop(lr=init_lr, clipnorm=1.0)
    vae.compile(vae_opt, loss=loss, metrics=[kl_div, likelihood])

    return vae, act_encoder, pose_encoder, decoder


def load_data(data_file, seq_length, seq_skip):
    db = P2DDataset(data_file, seq_length, seq_skip, gap=13, remove_head=True)
    train_X, _, train_A = db.get_pose_ds(train=True)
    train_A = np.argmax(train_A, axis=-1)
    val_X, _, val_A = db.get_pose_ds(train=False)
    val_A = np.argmax(val_A, axis=-1)
    return train_X, train_A, val_X, val_A, db


def train_model(train_X, train_A, val_X, val_A, db, args):
    # Make sure everything is the right shape
    assert val_X.shape[1] == args.seq_length, val_X.shape
    assert val_A.shape[1] == args.seq_length, val_A.shape
    assert val_A.ndim == 2, val_A.shape
    assert train_X.shape[1] == args.seq_length, train_X.shape
    assert train_A.shape[1] == args.seq_length, train_A.shape
    assert train_A.ndim == 2, val_A.shape
    for A in [train_A, val_A]:
        # should be vector of integer class labels
        # might want to add diversity check as well
        assert np.all((0 <= A) & (A < db.num_actions))

    _, _, pose_shape = train_X.shape
    assert train_X.shape[2] == pose_shape
    vae, act_encoder, pose_encoder, decoder = make_vae(pose_shape,
                                                       db.num_actions, args)

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
        # TODO: make this action-conditional! Rather uninteresting if it is not
        # action-conditional (although I can use an ipynb or s.th later to
        # explore if necessary).
        gen_relposes = decoder.predict(
            np.random.randn(poses_to_save, args.noise_dim))
        gen_poses = db.reconstruct_poses(gen_relposes)

        train_inds = np.random.permutation(len(train_X))[:poses_to_save]
        train_poses = db.reconstruct_poses(train_X[train_inds])

        val_inds = np.random.permutation(len(val_X))[:poses_to_save]
        val_poses = db.reconstruct_poses(val_X[val_inds])

        out_path = path.join(args.pose_dir, 'preds-epoch-%d.mat' % (epoch + 1))
        print('\nSaving samples to', out_path)
        np.savez(
            out_path,
            poses_gen=gen_poses,
            poses_train=train_poses,
            poses_val=val_poses,
            parents=db.parents)

    def model_paths(epoch, logs={}):
        model_path = path.join(
            args.model_dir, 'epoch-{epoch:02d}'.format(epoch=epoch))
        pose_encoder_path = model_path + '-pose-enc.h5'
        act_encoder_path = model_path + '-act-enc.h5'
        decoder_path = model_path + '-dec.h5'
        return pose_encoder_path, act_encoder_path, decoder_path

    def save_encoder_decoder(epoch, logs={}):
        epoch += args.extra_epoch
        pose_encoder_path, act_encoder_path, decoder_path = model_paths(epoch,
                                                                        logs)
        print('Saving pose encoder to %s' % pose_encoder_path)
        pose_encoder.save(act_encoder_path)
        print('Saving action encoder to %s' % act_encoder_path)
        act_encoder.save(act_encoder_path)
        print('Saving decoder to %s' % decoder_path)
        decoder.save(decoder_path)

    def save_state(epoch, logs={}):
        # Save all paths and arguments to file. Good for resumption of
        # training.
        epoch += args.extra_epoch
        pose_encoder_path, act_encoder_path, decoder_path = model_paths(epoch,
                                                                        logs)
        config_dest = args.config_path
        data = {
            'pose_encoder_path': pose_encoder_path,
            'act_encoder_path': act_encoder_path,
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
        sub_A = val_A[indices]

        # 'Extend' baseline
        ext_preds = sub_X[:, :1]
        ext_losses = np.mean(np.sum((sub_X - ext_preds)**2, axis=2), axis=0)
        del ext_preds

        # VAE baseline.
        K = 5
        vae_losses = np.zeros((sub_X.shape[0], sub_X.shape[1], K))
        for i in range(K):
            vae_preds = vae.predict({
                'reverse_poses': sub_X[:, ::-1],
                'reverse_actions': sub_A[:, ::-1]
            })
            vae_losses[:, :, i] = np.sum((sub_X - vae_preds)**2, axis=2)
        vae_mean_of_K = np.mean(np.mean(vae_losses, axis=2), axis=0)
        vae_best_of_K = np.mean(np.min(vae_losses, axis=2), axis=0)

        # Same as above, but we randomly shuffle inputs before doing
        # prediction. This will tell us whether we're actually learning a sane
        # embedding.
        random_losses = np.zeros((sub_X.shape[0], sub_X.shape[1], K))
        perm = np.random.permutation(len(sub_X))
        fake_X = sub_X[perm]
        fake_A = sub_A[perm]
        for i in range(K):
            random_preds = vae.predict({
                'reverse_poses': fake_X[:, ::-1],
                'reverse_actions': fake_A[:, ::-1]
            })
            random_losses[:, :, i] = np.sum((sub_X - random_preds)**2, axis=2)
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
        # patiences of 10 and 6 should ensure that changes are desynchronised
        ReduceLROnPlateau(patience=10),
        ScaleRamper(
            patience=6,
            schedule=args.kl_schedule,
            target_name='kl_scale',
            min_improvement=1,
            quantity='val_loss')
    ]
    vae.fit(
        # inputs are reversed so that output can be extracted in right order
        # TODO: figure out whether it's actually necessary to do this. Vastly
        # preferable not to, if possible to avoid it.
        {
            'reverse_poses': train_X[:, ::-1],
            'reverse_actions': train_A[:, ::-1]
        },
        train_X,
        validation_data=({
            'reverse_poses': val_X[:, ::-1],
            'reverse_actions': val_A[:, ::-1]
        }, val_X),
        shuffle=True,
        batch_size=args.batch_size,
        nb_epoch=1000,
        callbacks=cb_list)

    return vae, act_encoder, pose_encoder, decoder


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
    '--seq-length',
    type=int,
    dest='seq_length',
    default=16,
    help='length of input/output sequence')
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
# --feat-file can provide extra context for each frame in the encoder. The aim
# is to help the model learn better latent features.
# parser.add_argument('--feat-file', type=str, dest='feat_file', default=None,
#                     help='HDF5 file containing auxiliary, per-frame features')
parser.add_argument(
    '--kl-schedule',
    dest='kl_schedule',
    type=some_floats,
    default=(0.001, 0.01, 0.1, 0.3, 0.7, 1.0, ),
    help='KL divergence coefficient (try ramping this up)')
parser.add_argument(
    '--no-resume',
    action='store_false',
    dest='resume',
    default=True,
    help='stop automatic training resumption from checkpoint')


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
        args.pose_encoder_path = config['pose_encoder_path']
        args.act_encoder_path = config['act_encoder_path']
        args.decoder_path = config['decoder_path']
        args.extra_epoch = config['epoch']
    else:
        args.pose_encoder_path = args.act_encoder_path = args.decoder_path \
                                 = None
        args.extra_epoch = 0
        args._all_args = sys.argv[1:]
        assert args.data_file is not None, "Data file required if not resuming"

    return args


if __name__ == '__main__':
    args = load_args()

    print('Loading data')
    seq_length = args.seq_length
    train_X, train_A, val_X, val_A, db \
        = load_data(args.data_file, seq_length, args.seq_skip)
    print('Data loaded')

    print('Making directories')
    for dir_name in ['meta', 'pose', 'model', 'acc']:
        to_make = getattr(args, dir_name + '_dir')
        try:
            makedirs(to_make)
        except FileExistsError:
            pass

    train_model(train_X, train_A, val_X, val_A, db, args)
