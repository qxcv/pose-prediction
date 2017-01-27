#!/usr/bin/env python3

"""Try to jointly predict both future pose and future actions. Assumes a 2D
pose sequence (think Ikea dataset, Cooking Activities, Penn Action, etc.) and
performs normalisation on the fly."""

from argparse import ArgumentParser
import json
from os import path, makedirs
import sys

from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input, Lambda, RepeatVector
from keras.models import Model, load_model
from keras.objectives import categorical_crossentropy
from keras.optimizers import RMSprop

import numpy as np

import h5py


VAL_FRAC = 0.2


np.random.seed(2372143511)


def make_decoder(num_actions, seq_length, noise_dim):
    x = in_layer = Input(shape=(noise_dim,), name='dec_in')
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
    x = TimeDistributed(Dense(num_actions))(x)
    x = out_layer = Activation('softmax', name='dec_softmax')(x)

    decoder = Model(input=[in_layer], output=[out_layer])

    return decoder


def make_encoder(pose_size, seq_length, noise_dim):
    x = in_layer = Input(shape=(seq_length, pose_size,), name='enc_in')
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


def make_vae(pose_size, num_classes, args):
    seq_in_length = args.seq_in_length
    seq_out_length = args.seq_out_length
    noise_dim = args.noise_dim
    init_lr = args.init_lr

    encoder = make_encoder(pose_size, seq_in_length, noise_dim)
    if args.encoder_path:
        print('Reloading encoder weights from %s' % args.encoder_path)
        saved_encoder = load_model(args.encoder_path)
        copy_weights(saved_encoder, encoder)

    decoder = make_decoder(num_classes, seq_out_length, noise_dim)
    if args.decoder_path:
        print('Reloading decoder weights from %s' % args.decoder_path)
        saved_decoder = load_model(args.decoder_path)
        copy_weights(saved_decoder, decoder)

    # Sample from encoder's output distribution
    encoder_in = Input(shape=(seq_in_length, pose_size,))
    mean, var = encoder(encoder_in)
    log_std = Lambda(lambda var: 0.5 * K.log(var), output_shape=(noise_dim,), name='log_std')(var)
    std = Lambda(lambda var: K.sqrt(var), output_shape=(noise_dim,), name='std')(var)
    def make_noise(layers):  # noqa
        mean, std = layers
        noise = K.random_normal(shape=K.shape(std), mean=0., std=1.)
        return noise * std + mean
    latent = Lambda(make_noise, name='make_noise', output_shape=(noise_dim,))([mean, std])

    # Run latent variables through decoder
    decoder_out = decoder(latent)

    vae = Model(input=[encoder_in], output=[decoder_out])

    def kl_inner(l):
        # Mean loss is \|mu\|_2^2, std loss is tr(sig) - log(det(sig))
        mean, std, log_std = l
        return K.sum(K.square(mean) + std - log_std, axis=1) - noise_dim

    kl_loss = Lambda(kl_inner, output_shape=(None,), name='kl_loss')([mean, std, log_std])

    def kl_div(true, pred):
        return K.mean(kl_loss)

    def loss(y_true, y_pred):
        # TODO: Replace this with actual log likelihood to better fit VAE
        # formalism. Will also have to rescale the KL loss (which I think I
        # multiplied by two before).
        likelihood_loss = categorical_crossentropy(y_true, y_pred)

        return K.sum(likelihood_loss, axis=-1) + kl_loss

    vae_opt = RMSprop(lr=init_lr, clipnorm=1.0)
    vae.compile(vae_opt, loss=loss, metrics=[kl_div])

    return vae, encoder, decoder


def preprocess_sequence(poses, parents):
    """Preprocess a sequence of 2D poses to have more tractable representation.
    `parents` array is used to calculate output entries which are
    parent-relative joint locations. Note that standardisation will have to be
    performed later."""
    # Poses should be T*(XY)*J
    assert poses.ndim == 3, poses.shape
    assert poses.shape[1] == 2, poses.shape

    # Scale so that person roughly fits in 1x1 box at origin
    scale = (np.max(poses, axis=2) - np.min(poses, axis=2)).flatten().std()
    assert 1e-3 < scale < 1e4, scale
    offset = np.mean(np.mean(poses, axis=2), axis=0).reshape((1, 2, 1))
    norm_poses = (poses - offset) / scale

    # Compute actual data (relative offsets are easier to learn)
    relpose = np.zeros_like(norm_poses)
    relpose[0, 0, :] = 0
    # Head position records delta from previous frame
    relpose[1:, :, 0] = norm_poses[1:, :, 0] - norm_poses[:-1, :, 0]
    # Other norm_poses record delta from parents
    for jt in range(1, len(parents)):
        pa = parents[jt]
        relpose[:, :, jt] = norm_poses[:, :, jt] - norm_poses[:, :, pa]

    # Collapse in last two dimensions, interleaving X and Y coordinates
    shaped = relpose.reshape((relpose.shape[0], -1))

    return shaped


def load_data(data_file, seq_in_length, seq_out_length, seq_skip):
    total_seq_len = max(seq_in_length, seq_out_length)

    train_X_blocks = []
    train_Y_blocks = []
    val_X_blocks = []
    val_Y_blocks = []

    with h5py.File(data_file, 'r') as fp:
        parents = fp['/parents'].value
        num_actions = fp['/num_actions'].value.flatten()[0]

        vid_names = list(fp['seqs'])
        val_vid_list = list(vid_names)
        np.random.shuffle(val_vid_list)
        val_count = max(1, int(VAL_FRAC * len(val_vid_list)))
        val_vids = set(val_vid_list[:val_count])

        for vid_name in fp['seqs']:
            actions = fp['/seqs/' + vid_name + '/actions'].value
            one_hot_acts = np.zeros((len(actions), num_actions+1))
            one_hot_acts[(range(len(actions)), actions)] = 1

            poses = fp['/seqs/' + vid_name + '/poses'].value
            relposes = preprocess_sequence(poses, parents)

            assert len(relposes) == len(one_hot_acts)

            for i in range(len(relposes) - seq_skip * total_seq_len + 1):
                in_block = relposes[i:i+seq_skip*seq_in_length:seq_skip][::-1]
                out_block = one_hot_acts[i:i+seq_skip*seq_out_length:seq_skip]

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

    return train_X, train_Y, val_X, val_Y, mean, std


def train_model(train_X, train_Y, val_X, val_Y, args):
    # Make sure everything is the right shape
    assert val_X.shape[1] == args.seq_in_length, val_X.shape
    assert val_Y.shape[1] == args.seq_out_length, val_Y.shape
    assert train_X.shape[1] == args.seq_in_length, train_X.shape
    assert train_Y.shape[1] == args.seq_out_length, train_Y.shape

    _, _, in_shape = train_X.shape
    _, _, out_shape = train_Y.shape
    vae, encoder, decoder = make_vae(in_shape, out_shape, args)

    try:
        makedirs(args.pose_dir)
    except FileExistsError:
        pass
    try:
        makedirs(args.model_dir)
    except FileExistsError:
        pass

    # def sample_trajectories(epoch, logs={}):
    #     epoch += args.extra_epoch
    #     poses_to_save = args.poses_to_save
    #     gen_poses = decoder.predict(np.random.randn(poses_to_save, args.noise_dim))
    #     gen_poses = gen_poses * std + mean
    #     gen_poses = insert_junk_entries(gen_poses)

    #     train_inds = np.random.permutation(len(train_X))[:poses_to_save]
    #     train_poses = train_X[train_inds] * std + mean
    #     train_poses = insert_junk_entries(train_poses)

    #     val_inds = np.random.permutation(len(val_X))[:poses_to_save]
    #     val_poses = val_X[val_inds] * std + mean
    #     val_poses = insert_junk_entries(val_poses)

    #     out_path = path.join(args.pose_dir, 'preds-epoch-%d.mat' % (epoch + 1))
    #     print('\nSaving samples to', out_path)
    #     savemat(out_path, {
    #         'gen_poses': gen_poses,
    #         'train_poses': train_poses,
    #         'val_poses': val_poses
    #     })

    def model_paths(epoch, logs={}):
        model_path = path.join(args.model_dir, 'epoch-{epoch:02d}'.format(
            epoch=epoch
        ))
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
        extra_args = args._all_args
        config_dest = args.config_path
        data = {
            'encoder_path': encoder_path,
            'decoder_path': decoder_path,
            'args': extra_args,
            'epoch': epoch + 1
        }
        print('Saving config to', config_dest)
        with open(config_dest, 'w') as fp:
            json.dump(data, fp, indent=2)

    # def check_prediction_accuracy(epoch, logs={}):
    #     # Check prediction accuracy over entire validation dataset
    #     epoch += args.extra_epoch

    #     print('Calculating prediction accuracies')

    #     indices = np.random.permutation(len(val_X))[:1000]
    #     sub_X = val_X[indices]
    #     sub_Y = val_Y[indices]

    #     # 'Extend' baseline. Recall that val_X is time-reversed, so this
    #     # repeats the *last* pose in the input sequence (which may be in the
    #     # middle of the true output sequence).
    #     ext_preds = sub_X[:, :1]
    #     ext_losses = np.mean(np.sum((sub_Y - ext_preds) ** 2, axis=2), axis=0)
    #     del ext_preds

    #     # VAE baseline.
    #     K = 5
    #     vae_losses = np.zeros((sub_Y.shape[0], sub_Y.shape[1], K))
    #     for i in range(K):
    #         vae_preds = vae.predict(sub_X)
    #         vae_losses[:, :, i] = np.sum((sub_Y - vae_preds) ** 2, axis=2)
    #     vae_mean_of_K = np.mean(np.mean(vae_losses, axis=2), axis=0)
    #     vae_best_of_K = np.mean(np.min(vae_losses, axis=2), axis=0)

    #     # Same as above, but we randomly shuffle inputs before doing
    #     # prediction. This will tell us whether we're actually learning a sane
    #     # embedding.
    #     random_losses = np.zeros((sub_Y.shape[0], sub_Y.shape[1], K))
    #     fake_X = sub_X[np.random.permutation(len(sub_X))]
    #     for i in range(K):
    #         random_preds = vae.predict(fake_X)
    #         random_losses[:, :, i] = np.sum((sub_Y - random_preds) ** 2, axis=2)
    #     random_mean_of_K = np.mean(np.mean(random_losses, axis=2), axis=0)
    #     random_best_of_K = np.mean(np.min(random_losses, axis=2), axis=0)

    #     dest_path = path.join(args.acc_dir, 'epoch-%d.npz' % epoch)
    #     print('Saving accuracies to', dest_path)
    #     kwargs = {
    #         'vae_acc_mean_K': vae_mean_of_K,
    #         'vae_acc_best_K': vae_best_of_K,
    #         'random_acc_mean_K': random_mean_of_K,
    #         'random_acc_best_K': random_best_of_K,
    #         'ext_acc': ext_losses,
    #         'epoch': epoch,
    #         'K': K,
    #     }
    #     np.savez(dest_path, **kwargs)

    print('Training recurrent VAE')
    cb_list = [
        # LambdaCallback(on_epoch_end=sample_trajectories),
        # LambdaCallback(on_epoch_end=check_prediction_accuracy),
        LambdaCallback(on_epoch_end=save_encoder_decoder),
        LambdaCallback(on_epoch_end=save_state),
        ReduceLROnPlateau(patience=10)
    ]
    vae.fit(train_X, train_Y, validation_data=(val_X, val_Y),
            shuffle=True, batch_size=args.batch_size, nb_epoch=1000,
            callbacks=cb_list)

    return vae, encoder, decoder


# TODO: Things which are probably beneficial to change over time:
# - Coefficient on KL divergence term

parser = ArgumentParser()
parser.add_argument('--lr', type=float, dest='init_lr', default=0.0001,
                    help='initial learning rate')
parser.add_argument('--work-dir', type=str, dest='work_dir', default='./seq-act-vae',
                    help='parent directory to store state')
parser.add_argument('--seq-in-length', type=int, dest='seq_in_length', default=16,
                    help='length of input sequence')
parser.add_argument('--seq-out-length', type=int, dest='seq_out_length', default=16,
                    help='length of sequence to predict (may overlap)')
# parser.add_argument('--save-poses', type=int, dest='poses_to_save', default=32,
#                     help='number of sample poses to save at each epoch')
parser.add_argument('--seq-skip', type=int, dest='seq_skip', default=3,
                    help='factor by which to temporally downsample data')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=64,
                    help='size of training batch')
parser.add_argument('--noise-dim', type=int, dest='noise_dim', default=64,
                    help='number of latent variables')
parser.add_argument('--data-file', type=str, dest='data_file', default=None,
                    help='HDF5 file containing poses')
parser.add_argument('--no-resume', action='store_false', dest='resume', default=True,
                    help='stop automatic training resumption from checkpoint')


def add_extra_paths(args):
    wd = args.work_dir
    args.meta_dir = path.join(wd, 'meta')
    args.log_dir = path.join(wd, 'logs')
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
        args._all_args = arg_list
        add_extra_paths(args)
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
    train_X, train_Y, val_X, val_Y, mean, std \
        = load_data(args.data_file, args.seq_in_length, args.seq_out_length, args.seq_skip)
    print('Data loaded')

    print('Making directories')
    for dir_name in ['meta', 'log', 'pose', 'model', 'acc']:
        to_make = getattr(args, dir_name + '_dir')
        try:
            makedirs(to_make)
        except FileExistsError:
            pass

    std_mean_path = path.join(args.meta_dir, 'std_mean.json')
    print('Saving mean/std to %s' % std_mean_path)
    with open(std_mean_path, 'w') as fp:
        to_dump = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        json.dump(to_dump, fp)

    model = train_model(train_X, train_Y, val_X, val_Y, args)
