#!/usr/bin/env python3
"""Generate a sequence of poses with convolutional generator and
discriminator."""

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    RepeatVector, Input, Dropout, LeakyReLU, Convolution1D, Flatten, \
    BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils.generic_utils import Progbar
import numpy as np
import re
from glob import glob
from os import path, mkdir, makedirs
from scipy.io import savemat

from common import GOOD_MOCAP_INDS, insert_junk_entries

np.random.seed(2372143511)

WEIGHTS_PATH = './best-conv-gan-weights.h5'
SEQ_LENGTH = 32
SEQ_NOISE_PAD = 7
NOISE_DIM = 30
BATCH_SIZE = 16
K = 5


def make_generator(pose_size):
    x = in_layer = Input(shape=(SEQ_LENGTH, NOISE_DIM))
    x = Convolution1D(500, 3, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(500, 3, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(500, 3, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(500, 3, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(500, 3, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(500, 3, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = out_layer = Convolution1D(pose_size, 3, border_mode='valid')(x)

    model = Model(input=[in_layer], output=[out_layer])

    return model


def make_discriminator(pose_size):
    in_shape = (SEQ_LENGTH, pose_size)

    x = in_layer = Input(shape=in_shape)
    x = Convolution1D(128, 8, border_mode='valid')(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution1D(128, 8, border_mode='valid')(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution1D(1, 8, border_mode='valid')(x)
    x = Flatten()(x)
    out_layer = Activation('sigmoid')(x)

    model = Model(input=[in_layer], output=[out_layer])

    return model


class GANTrainer:
    noise_dim = NOISE_DIM
    seq_length = SEQ_LENGTH
    seq_pad = SEQ_NOISE_PAD

    def __init__(self, pose_size, d_lr=0.0002, g_lr=0.00002):
        self.discriminator = make_discriminator(pose_size)
        # Copy is read-only; it doesn't get compiled
        self.discriminator_copy = make_discriminator(pose_size)
        self.discriminator_copy.trainable = False
        disc_opt = Adam(lr=d_lr, beta_1=0.5)
        self.discriminator.compile(disc_opt, 'binary_crossentropy',
                                   metrics=['binary_accuracy'])

        self.generator = make_generator(pose_size)
        self.generator.compile('sgd', 'mae')

        nested = Sequential()
        nested.add(self.generator)
        nested.add(self.discriminator_copy)
        gen_opt = Adam(lr=g_lr, beta_1=0.5)
        nested.compile(gen_opt, 'binary_crossentropy',
                       metrics=['binary_accuracy'])
        self.nested_generator = nested

        self.num_disc_steps = 0
        self.num_gen_steps = 0

    def update_disc_copy(self):
        """Copy weights from real discriminator over to nested one. This skirts
        a lot of Keras issues with nested, shared models."""
        source = self.discriminator
        dest = self.discriminator_copy

        assert len(source.layers) == len(dest.layers)
        for dest_layer, source_layer in zip(dest.layers, source.layers):
            dest_layer.set_weights(source_layer.get_weights())

    def make_noise(self, num):
        """Input noise for generator"""
        return np.random.randn(num, self.seq_length + 2 * self.seq_pad,
                               self.noise_dim)

    def gen_train_step(self, batch_size):
        """Train the generator to fool the discriminator."""
        self.discriminator.trainable = False
        labels = [1] * batch_size
        noise = self.make_noise(batch_size)
        self.update_disc_copy()
        self.num_gen_steps += 1
        pre_weights = self.discriminator_copy.get_weights()
        rv = self.nested_generator.train_on_batch(noise, labels)
        post_weights = self.discriminator_copy.get_weights()
        # The next assertion fails with batch norm. I don't know how to stop
        # those layers from updating :(
        assert all(np.all(a == b) for a, b in zip(pre_weights, post_weights))
        return rv

    def disc_train_step(self, true_batch):
        """Get some true poses and train discriminator to distinguish them from
        generated poses."""
        self.discriminator.trainable = True
        poses = self.generate_poses(len(true_batch))
        labels = np.array([1] * len(true_batch) + [0] * len(poses))
        data = np.concatenate([true_batch, poses])
        self.num_disc_steps += 1
        # Get back loss
        return self.discriminator.train_on_batch(data, labels)

    def disc_val(self, val_data, batch_size):
        """Validate discriminator by checking whether it can spot fakes."""
        fakes = self.generate_poses(len(val_data))
        labels = np.array([1] * len(val_data) + [0] * len(fakes))
        data = np.concatenate([val_data, fakes])
        return self.discriminator.evaluate(data, labels,
                                           batch_size=batch_size)

    def gen_val(self, num_poses, batch_size):
        """Validate generator by figuring out how good it is at fooling
        discriminator (closely related to discriminator step; just helps us
        break down accuracy a bit)."""
        noise = self.make_noise(num_poses)
        labels = [1] * num_poses
        self.update_disc_copy()
        rv = self.nested_generator.evaluate(noise, labels,
                                            batch_size=batch_size)
        self.update_disc_copy()
        return rv

    def generate_poses(self, num, batch_size=BATCH_SIZE):
        """Generate some fixed number of poses. Useful for both generator and
        discriminator training."""
        return self.generator.predict(self.make_noise(num),
                                      batch_size=batch_size)

    def save(self, dest_dir):
        """Save generator and discriminator to some path"""
        try:
            makedirs(dest_dir)
        except FileExistsError:
            pass
        suffix = '-%d-%d.h5' % (self.num_gen_steps, self.num_disc_steps)
        gen_path = path.join(dest_dir, 'gen' + suffix)
        disc_path = path.join(dest_dir, 'disc' + suffix)
        self.discriminator.save(disc_path)
        self.generator.save(gen_path)


def train_model(train_X, val_X, mu, sigma):
    assert train_X.ndim == 3, train_X.shape
    total_X, time_steps, out_shape = train_X.shape
    trainer = GANTrainer(out_shape)
    epochs = 0

    # GAN predictions will be put in here
    try:
        mkdir('gan-conv-out')
    except FileExistsError:
        pass

    print('Training generator')

    while True:
        copy_X = train_X.copy()
        np.random.shuffle(copy_X)
        total_X, _, _ = copy_X.shape
        to_fetch = BATCH_SIZE // 2
        epochs += 1
        print('Epoch %d' % epochs)
        bar = Progbar(total_X)
        bar.update(0)
        epoch_fetched = 0

        while epoch_fetched < total_X:
            # Fetch some ground truth to train the discriminator
            for i in range(K):
                if epoch_fetched >= total_X:
                    break
                fetched = copy_X[epoch_fetched:epoch_fetched+to_fetch]
                dloss, dacc = trainer.disc_train_step(fetched)
                epoch_fetched += len(fetched)
                bar.update(epoch_fetched, values=[
                    ('d_loss', dloss), ('d_acc', dacc)
                ])

            # Train the generator (don't worry about loss)
            trainer.gen_train_step(BATCH_SIZE)

        # End of an epoch, so let's validate models (doesn't work so great,
        # TBH)
        print('\nValidating')
        disc_loss, disc_acc = trainer.disc_val(val_X, BATCH_SIZE)
        gen_loss, gen_acc = trainer.gen_val(100, BATCH_SIZE)
        print('\nDisc loss/acc:   %g/%g' % (disc_loss, disc_acc))
        print('Gen loss/acc:    %g/%g' % (gen_loss, gen_acc))

        # Also save some predictions so that we can monitor training
        print('Saving predictions')
        poses = trainer.generate_poses(16) * sigma + mean
        poses = insert_junk_entries(poses)
        savemat('gan-conv-out/gan-conv-preds-epoch-%d.mat' % epochs, {'poses': poses})

        # Sometimes we save a model
        if not (epochs - 1) % 5:
            dest_dir = 'saved-conv-gans/'
            print('Saving model to %s' % dest_dir)
            trainer.save(dest_dir)


def prepare_file(filename):
    poses = np.loadtxt(filename, delimiter=',')
    assert poses.ndim == 2 and poses.shape[1] == 99, poses.shape

    zero_inds, = np.nonzero((poses != 0).any(axis=0))
    assert (zero_inds == GOOD_MOCAP_INDS).all(), zero_inds
    poses = poses[:, GOOD_MOCAP_INDS]

    seqs = []
    end = len(poses) - SEQ_LENGTH + 1
    # TODO: May make sense to have a bigger overlap here
    step = max(1, min(SEQ_LENGTH // 2, 50))
    for start in range(0, end, step):
        seqs.append(poses[start:start+SEQ_LENGTH])

    return np.stack(seqs)


def is_valid(data):
    return np.isfinite(data).all()


def load_data():
    fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')

    filenames = glob('h36m-3d-poses/expmap_*.txt.gz')

    train_X_blocks = []
    test_X_blocks = []

    for filename in filenames:
        base = path.basename(filename)
        meta = fnre.match(base).groupdict()
        subj_id = int(meta['subject'])

        X = prepare_file(filename)

        if subj_id == 5:
            # subject 5 is for testing
            test_X_blocks.append(X)
        else:
            train_X_blocks.append(X)

    train_X = np.concatenate(train_X_blocks, axis=0)
    test_X = np.concatenate(test_X_blocks, axis=0)

    N, T, D = train_X.shape

    mean = train_X.reshape((N*T, D)).mean(axis=0).reshape((1, 1, -1))
    std = train_X.reshape((N*T, D)).std(axis=0).reshape((1, 1, -1))

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    assert is_valid(train_X)
    assert is_valid(test_X)

    return train_X, test_X, mean, std


if __name__ == '__main__':
    print('Loading data')
    train_X, val_X, mean, std = load_data()
    print('Data loaded')

    model = train_model(train_X, val_X, mean, std)
