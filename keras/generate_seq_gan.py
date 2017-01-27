#!/usr/bin/env python3
"""Generate a sequence of poses. Generator and discriminator both based on
ERD."""

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, TimeDistributed, LSTM, \
    Input, LeakyReLU, Bidirectional, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils.generic_utils import Progbar
import numpy as np
import re
from glob import glob
from os import path, makedirs
from scipy.io import savemat
from multiprocessing import Pool

from common import GOOD_MOCAP_INDS, insert_junk_entries

np.random.seed(2372143511)

WORK_DIR = './seq-gan/'
MODEL_DIR = path.join(WORK_DIR, 'models')
SAMPLE_DIR = path.join(WORK_DIR, 'samples')
LOG_DIR = path.join(WORK_DIR, 'logs')
SEQ_LENGTH = 32
NOISE_DIM = 30
BATCH_SIZE = 64
# Factor by which to temporally downsample motion sequences. Makes motion
# "slower", from network PoV.
SEQ_SKIP = 3
K = 5


def make_generator(pose_size):
    x = in_layer = Input(shape=(SEQ_LENGTH, NOISE_DIM))
    x = TimeDistributed(Dense(500))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Dense(500))(x)
    x = Bidirectional(LSTM(1000, return_sequences=True))(x)
    x = Bidirectional(LSTM(1000, return_sequences=True))(x)
    x = TimeDistributed(Dense(500))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # XXX: Why is there a 100 unit layer here? Is it actually from the ERD
    # paper?
    # x = TimeDistributed(Dense(100))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    out_layer = TimeDistributed(Dense(pose_size))(x)

    model = Model(input=[in_layer], output=[out_layer])

    return model


def make_discriminator(pose_size):
    in_shape = (SEQ_LENGTH, pose_size)

    x = in_layer = Input(shape=in_shape)
    x = TimeDistributed(Dense(128))(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(128))(x)
    x = Bidirectional(LSTM(500, return_sequences=True))(x)
    x = Bidirectional(LSTM(500))(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1)(x)
    out_layer = Activation('sigmoid')(x)

    model = Model(input=[in_layer], output=[out_layer])

    return model


class GANTrainer:
    noise_dim = NOISE_DIM
    seq_length = SEQ_LENGTH

    def __init__(self, pose_size, d_lr=0.000001, g_lr=0.000001):
        self.discriminator = make_discriminator(pose_size)
        # Copy is read-only; it doesn't get compiled
        self.discriminator_copy = make_discriminator(pose_size)
        self.discriminator_copy.trainable = False
        disc_opt = RMSprop(lr=d_lr, clipnorm=1.0)
        self.discriminator.compile(disc_opt, 'binary_crossentropy',
                                   metrics=['binary_accuracy'])

        self.generator = make_generator(pose_size)
        self.generator.compile('sgd', 'mae')

        nested = Sequential()
        nested.add(self.generator)
        nested.add(self.discriminator_copy)
        gen_opt = RMSprop(lr=g_lr, clipnorm=1.0)
        nested.compile(gen_opt, 'binary_crossentropy',
                       metrics=['binary_accuracy'])
        self.nested_generator = nested

        self.num_disc_steps = 0
        self.num_gen_steps = 0

        # XXX: Need to set validation_data correctly
        # self.nested_tb = TensorBoard(path.join(LOG_DIR, 'nest-gen'),
        #                              histogram_freq=1)
        # self.nested_tb._set_model(self.nested_generator.model)
        # self.disc_tb = TensorBoard(path.join(LOG_DIR, 'disc'),
        #                            histogram_freq=1)
        # self.disc_tb._set_model(self.discriminator)

    _tb_epochs = 0

    def run_tb_callbacks(self):
        self._tb_epochs += 1
        # self.nested_tb.on_epoch_end(self._tb_epochs)
        # self.disc_tb.on_epoch_end(self._tb_epochs)

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
        return np.random.randn(num, self.seq_length, self.noise_dim)

    def gen_train_step(self, batch_size):
        """Train the generator to fool the discriminator."""
        self.discriminator.trainable = False
        labels = [1] * batch_size
        noise = self.make_noise(batch_size)
        self.update_disc_copy()
        self.num_gen_steps += 1
        # pre_weights = self.discriminator_copy.get_weights()
        rv = self.nested_generator.train_on_batch(noise, labels)
        # post_weights = self.discriminator_copy.get_weights()
        # The next assertion fails with batch norm. I don't know how to stop
        # those layers from updating :(
        # assert all(np.all(a == b) for a, b in zip(pre_weights, post_weights))
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
        rv = self.discriminator.evaluate(data, labels,
                                         batch_size=batch_size)

        # Calling .fit() stores .validation_data on self.discriminator. The
        # TensorBoard callback can then use that to make an activation
        # histogram (or whatever it does). Need to pick first [:100] or
        # TensorFlow runs out of memory :P
        fit_indices = np.random.permutation(len(data))[:100]
        fit_data = (data[fit_indices], labels[fit_indices])
        self.discriminator.fit(*fit_data, batch_size=batch_size, nb_epoch=1,
                               verbose=0, validation_data=fit_data)

        return rv

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

        # Same trick we pull in disc_val
        fit_data = (noise[:100], labels[:100])
        self.nested_generator.fit(*fit_data, batch_size=batch_size, nb_epoch=1,
                                  verbose=0, validation_data=fit_data)

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


def random_subset(array, count):
    """Choose a random set of 'count' rows."""
    indices = np.random.permutation(len(array))[:count]
    return array[indices]


def train_model(train_X, val_X, mu, sigma):
    assert train_X.ndim == 3, train_X.shape
    total_X, time_steps, out_shape = train_X.shape
    trainer = GANTrainer(out_shape)
    epochs = 0

    # GAN predictions will be put in here
    try:
        makedirs(SAMPLE_DIR)
    except FileExistsError:
        pass

    print('Training generator')

    while True:
        # Only use a small subset of data in each batch (20K points)
        copy_X = random_subset(train_X, 20000)
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
        # Again, use a small subset of 5K points
        to_val = random_subset(val_X, 5000)
        disc_loss, disc_acc = trainer.disc_val(to_val, BATCH_SIZE)
        gen_loss, gen_acc = trainer.gen_val(100, BATCH_SIZE)
        print('\nDisc loss/acc:   %g/%g' % (disc_loss, disc_acc))
        print('Gen loss/acc:    %g/%g' % (gen_loss, gen_acc))

        # Also save some predictions so that we can monitor training
        print('Saving predictions')
        gen_poses = trainer.generate_poses(16) * sigma + mean
        gen_poses = insert_junk_entries(gen_poses)

        def sample_real(real, count=16, sigma=sigma, mean=mean):
            indices = np.random.permutation(len(real))[:count]
            real_poses = real[indices] * sigma + mean
            return insert_junk_entries(real_poses)

        sampled_train_poses = sample_real(copy_X)
        sampled_val_poses = sample_real(to_val)
        dest_path = path.join(SAMPLE_DIR, 'epoch-%d.mat' % epochs)
        print('Saving samples to', dest_path)
        savemat(dest_path, {
            'gen_poses': gen_poses,
            'train_poses': sampled_train_poses,
            'val_poses': sampled_val_poses
        })

        # Sometimes we save a model
        if not (epochs - 1) % 5:
            print('Saving model to %s' % MODEL_DIR)
            trainer.save(MODEL_DIR)

        print('Running TensorBoard callbacks')
        trainer.run_tb_callbacks()


def prepare_file(filename, seq_length, seq_skip):
    poses = np.loadtxt(filename, delimiter=',')
    assert poses.ndim == 2 and poses.shape[1] == 99, poses.shape

    zero_inds, = np.nonzero((poses != 0).any(axis=0))
    assert (zero_inds == GOOD_MOCAP_INDS).all(), zero_inds
    poses = poses[:, GOOD_MOCAP_INDS]

    seqs = []
    true_length = seq_length * seq_skip
    end = len(poses) - true_length + 1
    # TODO: Might not want to overlap sequences so much. Then again, it may not
    # matter given that I'm shuffling anyway
    for start in range(end):
        seqs.append(poses[start:start+true_length:seq_skip])

    return np.stack(seqs)


def is_valid(data):
    return np.isfinite(data).all()


_fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')


def _mapper(arg):
    """Worker to load data in parallel"""
    filename, seq_length, seq_skip = arg
    base = path.basename(filename)
    meta = _fnre.match(base).groupdict()
    subj_id = int(meta['subject'])
    X = prepare_file(filename, seq_length, seq_skip)

    return subj_id, filename, X


def load_data(seq_length=SEQ_LENGTH, seq_skip=SEQ_SKIP, val_subj_5=True):
    filenames = glob('h36m-3d-poses/expmap_*.txt.gz')

    train_X_blocks = []
    test_X_blocks = []

    if not val_subj_5:
        # Need to make a pool of val_filenames
        all_inds = np.random.permutation(len(filenames))
        val_count = int(0.2*len(all_inds)+1)
        val_inds = all_inds[:val_count]
        fn_arr = np.array(filenames)
        val_filenames = set(fn_arr[val_inds])

    print('Spawning pool')
    with Pool() as pool:
        fn_seq = ((fn, seq_length, seq_skip) for fn in filenames)
        for subj_id, filename, X in pool.map(_mapper, fn_seq):
            if val_subj_5:
                is_val = subj_id == 5
            else:
                is_val = filename in val_filenames
            if is_val:
                # subject 5 is for testing
                test_X_blocks.append(X)
            else:
                train_X_blocks.append(X)

    # Memory usage is right on the edge of what small machines are capable of
    # handling here, so I'm being careful to delete large unneeded structures.
    train_X = np.concatenate(train_X_blocks, axis=0)
    del train_X_blocks
    test_X = np.concatenate(test_X_blocks, axis=0)
    del test_X_blocks

    N, T, D = train_X.shape

    reshaped = train_X.reshape((N*T, D))
    mean = reshaped.mean(axis=0).reshape((1, 1, -1))
    std = reshaped.std(axis=0).reshape((1, 1, -1))
    del reshaped

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
