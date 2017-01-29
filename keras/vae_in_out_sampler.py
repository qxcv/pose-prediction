#!/usr/bin/env python3
"""Script to sample some pose subsequences from specified files, then run them
through an encoder/decoder to get a reconstructed pose sequence. Will write
pairs (original/reconstruction) to some output file which is easy to read. """

from argparse import ArgumentParser
from os import path, makedirs
from sys import stderr

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Lambda
import keras.backend as K

from common import insert_junk_entries
from generate_seq_vae import make_encoder, make_decoder, copy_weights, \
    load_data


def load_models(model_dir, epoch):
    enc_path = path.join(model_dir, 'models', 'epoch-%02d-enc.h5' % epoch)
    saved_encoder = load_model(enc_path)

    dec_path = path.join(model_dir, 'models', 'epoch-%02d-dec.h5' % epoch)
    saved_decoder = load_model(dec_path)

    return saved_encoder, saved_decoder


def make_vae(pose_size, seq_length, model_dir, epoch):
    saved_encoder, saved_decoder = load_models(model_dir, epoch)
    noise_dim = saved_decoder.input_shape[1]

    print('Making encoder and copying weights')
    encoder = make_encoder(pose_size, seq_length, noise_dim)
    copy_weights(saved_encoder, encoder)

    print('Making decoder and copying weights')
    decoder = make_decoder(pose_size, seq_length, noise_dim)
    copy_weights(saved_decoder, decoder)

    print('Making full model')
    encoder_in = Input(shape=(
        seq_length,
        pose_size, ))
    mean, var = encoder(encoder_in)
    std = Lambda(lambda var: K.sqrt(var))(var)

    def make_noise(layers):  # noqa
        mean, std = layers
        noise = K.random_normal(shape=K.shape(std), mean=0., std=1.)
        return noise * std + mean

    latent = Lambda(make_noise)([mean, std])
    decoder_out = decoder(latent)
    vae = Model(input=[encoder_in], output=[decoder_out])

    return vae


def take_samples(X, vae, mean, std, num_samples, num_alts, out_dir):
    sample_inds = np.random.permutation(len(X))[:num_samples]
    sub_X = X[sample_inds]
    alt_preds = []

    for i in range(num_alts):
        print('Sampling alt %d/%d' % (i, num_alts))
        preds = vae.predict(sub_X)
        preds = preds * std + mean
        preds = insert_junk_entries(preds)
        alt_preds.append(preds)

    print('Sampling done, writing data')
    for s in range(num_samples):
        s_out_dir = path.join(out_dir, 'sample-%d' % s)
        try:
            makedirs(s_out_dir)
        except FileExistsError:
            pass

        # Write original
        orig_path = path.join(s_out_dir, 'orig.txt')
        orig = sub_X[s]
        np.savetxt(orig_path, orig, fmt='%f', delimiter=', ')

        # Write alternatives
        for a in range(num_alts):
            alt_path = path.join(s_out_dir, 'alt-%d.txt' % a)
            alt = alt_preds[a][s]
            np.savetxt(alt_path, alt)


parser = ArgumentParser()
parser.add_argument('--epoch', default=1, help='epoch number of saved model')
parser.add_argument(
    '--model-dir',
    dest='model_dir',
    default='seq-vae',
    help='path to saved training data')
parser.add_argument(
    '--seq-length',
    dest='seq_length',
    type=int,
    default=80,
    help='length of sequence to generate')
parser.add_argument(
    '--num-train-samples',
    dest='num_train_samples',
    type=int,
    default=100,
    help='number of training sequences to sample')
parser.add_argument(
    '--num-val-samples',
    dest='num_val_samples',
    type=int,
    default=100,
    help='number of valing sequences to sample')
parser.add_argument(
    '--num-alts',
    dest='num_alts',
    type=int,
    default=3,
    help='alternative completions per sequence')
parser.add_argument(
    '--seq-skip',
    dest='seq_skip',
    type=int,
    default=3,
    help='alternative completions per sequence')
parser.add_argument(
    '--out-dir',
    dest='out_dir',
    default='vae-samples',
    help='alternative completions per sequence')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.epoch <= 5:
        print(
            'Using model trained at epoch %d (!!). Use --epoch to change '
            'this, if desired' % args.epoch,
            file=stderr)

    print('Loading data')
    train_X, val_X, mean, std = load_data(
        args.seq_length, args.seq_skip, val_subj_5=False)
    pose_size = train_X.shape[2]

    print('Building models')
    vae = make_vae(pose_size, args.seq_length, args.model_dir, args.epoch)

    print('Taking training samples')
    train_out = path.join(args.out_dir, 'train')
    take_samples(train_X, vae, mean, std, args.num_train_samples,
                 args.num_alts, train_out)

    print('Taking validation samples')
    val_out = path.join(args.out_dir, 'val')
    take_samples(val_X, vae, mean, std, args.num_val_samples, args.num_alts,
                 val_out)

    print('Done! Everything saved to %s' % args.out_dir)
