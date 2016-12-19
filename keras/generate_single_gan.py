#!/usr/bin/env python3

"""GAN which generates a single pose at a time. Preparation for full GAN. Might
also be able to turn it into an InfoGAN :)"""

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Activation, Dropout, \
    GaussianNoise, LeakyReLU
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import re
from glob import glob
from os import path, mkdir
from scipy.io import savemat

from common import CUSTOM_OBJECTS, GOOD_MOCAP_INDS, insert_junk_entries

np.random.seed(2372143511)

WEIGHTS_PATH = './best-single-gan-weights.h5'
# Use NOISE_DIM-dimensional independent Gaussian noise
NOISE_DIM = 20
BATCH_SIZE = 64
# Discriminator train batches for each generator batch. Ideally, we keep the
# discriminator a long way ahead of the generator.
K = 5


def make_generator(pose_size):
    model = Sequential()
    model.add(Dense(256, input_dim=NOISE_DIM))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1000))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1000))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1000))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(pose_size))
    model.add(Activation('tanh'))

    return model


def make_discriminator(pose_size):
    # Hopefully this much narrower network will not be able to overfit to ~2MM
    # poses. That should, in turn, prevent the generator from overfitting.
    # I'm using dropout instead of BatchNormalization because there's some bug
    # causing BatchNormalization to fail when you put it in a model which is
    # embedded downstream of something else.
    model = Sequential()
    # model.add(GaussianNoise(0.01, input_shape=(pose_size,)))
    model.add(Dense(128, input_shape=(pose_size,)))
    model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


class GANTrainer:
    noise_dim = NOISE_DIM

    def __init__(self, pose_size, d_lr=0.0002, g_lr=0.0002, adam_b1=0.5):
        # Compile individual models
        # Opt parameters taken from DCGAN paper
        self.discriminator = make_discriminator(pose_size)
        # Copy is read-only; it doesn't get compiled
        self.discriminator_copy = make_discriminator(pose_size)
        self.discriminator_copy.trainable = False
        disc_opt = Adam(lr=d_lr, beta_1=adam_b1)
        self.discriminator.compile(disc_opt, 'binary_crossentropy',
                                   metrics=['binary_accuracy'])

        self.generator = make_generator(pose_size)
        # AFAIK this doesn't really matter
        self.generator.compile('sgd', 'mae')

        # Nested model will be useful for training generator
        nested = Sequential()
        nested.add(self.generator)
        nested.add(self.discriminator_copy)
        gen_opt = Adam(lr=g_lr, beta_1=adam_b1)
        nested.compile(gen_opt, 'binary_crossentropy',
                       metrics=['binary_accuracy'])
        self.nested_generator = nested

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
        return np.random.randn(num, self.noise_dim)

    def gen_train_step(self, batch_size):
        """Train the generator to fool the discriminator."""
        self.discriminator.trainable = False
        labels = [1] * batch_size
        noise = self.make_noise(batch_size)
        self.update_disc_copy()
        return self.nested_generator.train_on_batch(noise, labels)

    def disc_train_step(self, true_batch):
        """Get some true poses and train discriminator to distinguish them from
        generated poses."""
        self.discriminator.trainable = True
        poses = self.generate_poses(len(true_batch))
        labels = np.array([1] * len(true_batch) + [0] * len(poses))
        data = np.concatenate([true_batch, poses])
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
        return self.nested_generator.evaluate(noise, labels,
                                              batch_size=batch_size)

    def generate_poses(self, num):
        """Generate some fixed number of poses. Useful for both generator and
        discriminator training."""
        return self.generator.predict_on_batch(self.make_noise(num))


def train_model(train_X, val_X, mu, sigma):
    assert train_X.ndim == 2, train_X.ndim
    total_X, out_shape = train_X.shape
    trainer = GANTrainer(train_X.shape[1])
    epochs = 0

    # GAN predictions will be put in here
    try:
        mkdir('gan-out')
    except FileExistsError:
        pass

    print('Training generator')

    while True:
        copy_X = train_X.copy()
        np.random.shuffle(copy_X)
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
        gen_loss, gen_acc = trainer.gen_val(1000, BATCH_SIZE)
        print('\nDisc loss/acc:   %g/%g' % (disc_loss, disc_acc))
        print('Gen loss/acc:    %g/%g' % (gen_loss, gen_acc))

        # Also save some predictions so that we can monitor training
        print('Saving predictions')
        poses = trainer.generate_poses(256) * sigma + mean
        poses = insert_junk_entries(poses)
        savemat('gan-out/gan-preds-epoch-%d.mat' % epochs, {'poses': poses})


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


def load_data():
    fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')

    filenames = glob('h36m-3d-poses/expmap_*.txt.gz')

    train_X_blocks = []
    test_X_blocks = []

    for filename in filenames:
        base = path.basename(filename)
        meta = fnre.match(base).groupdict()
        subj_id = int(meta['subject'])

        X = prepare_data_file(filename)

        if subj_id == 5:
            # subject 5 is for testing
            test_X_blocks.append(X)
        else:
            train_X_blocks.append(X)

    train_X = np.concatenate(train_X_blocks, axis=0)
    test_X = np.concatenate(test_X_blocks, axis=0)

    mean = train_X.mean(axis=0).reshape((1, -1))
    std = train_X.std(axis=0).reshape((1, -1))

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    np.random.shuffle(train_X)
    np.random.shuffle(test_X)

    assert is_valid(train_X)
    assert is_valid(test_X)
    assert is_valid(mean)
    assert is_valid(std)

    return train_X, test_X,  mean, std


if __name__ == '__main__':
    print('Loading data')
    train_X, val_X, mean, std = load_data()
    print('Data loaded')

    model = None
    try:
        print('Loading model')
        model = load_model(WEIGHTS_PATH, CUSTOM_OBJECTS)
    except OSError:
        print('Load failed, building model anew')
    if model is None:
        model = train_model(train_X, val_X, mean, std)
