"""Code shared between recurrent and non-recurrent pose prediction models."""

from collections import OrderedDict
from glob import glob
import itertools
from os import path
import re

import numpy as np

from scipy.stats import norm

from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Layer

# here because I can't be bothered updating other code to point to h36m_loader
from h36m_loader import GOOD_MOCAP_INDS, insert_junk_entries  # flake8: noqa

PA = [0, 0, 1, 2, 3, 1, 5, 6]
THRESHOLDS = [0.05, 0.15, 0.25, 0.5, 0.8]
# Dictionary to pass to load_model, etc., to find custom objectives and layers
CUSTOM_OBJECTS = {}
# Standard deviations for noise
NOISE_SCHEDULE = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
# Correspond to horizons of 80, 160, 240, 320, 400, 480, 560ms
PREDICT_OFFSETS = [4, 8, 12, 16, 20, 24, 28]


def scrape_sequences(model, data, data_mean, data_std, num_to_scrape,
                     seq_length):
    sel_indices = np.random.choice(
        np.arange(data.shape[0]), size=num_to_scrape, replace=False)

    assert data.ndim == 3
    assert data.shape[1] == seq_length - 1

    train_k = data.shape[1] // 2
    all_preds = np.zeros((num_to_scrape, ) + data.shape[1:])

    for pi, ind in enumerate(sel_indices):
        seq = data[ind]
        model.reset_states()
        preds = np.zeros(data.shape[1:])

        for i in range(train_k):
            preds[i, :] = model.predict(seq[i].reshape((1, 1, -1)))

        for i in range(train_k, data.shape[1]):
            preds[i, :] = model.predict(preds[i - 1, :].reshape((1, 1, -1)))

        all_preds[pi, :, :] = preds * data_std[ind] + data_mean[ind]

    return all_preds


def get_offset_losses(model,
                      test_X,
                      test_Y,
                      test_means,
                      test_stds,
                      to_sample=100):
    """Try to imitate ERD paper quant eval: compare true values of Y (after
    adding mean/std) with fake values.

    I really don't know what the ERD paper was actually using for quant eval,
    though, so this will probably be some way from the right result :/"""
    sel_indices = np.random.choice(
        np.arange(test_X.shape[0]), size=to_sample, replace=False)
    # Use the first half of the sequence as a prefix
    train_k = test_X.shape[1] // 2
    losses = {}

    for pi, ind in enumerate(sel_indices):
        seq = test_X[ind]
        model.reset_states()
        preds = np.zeros(test_X.shape[1:])

        for i in range(train_k):
            preds[i, :] = model.predict(seq[i].reshape((1, 1, -1)))

        for i in range(train_k, test_X.shape[1]):
            preds[i, :] = model.predict(preds[i - 1, :].reshape((1, 1, -1)))

        mean, std = test_means[ind, 0], test_stds[ind, 0]
        for offset in PREDICT_OFFSETS:
            i = train_k + offset
            pred = preds[i] * std + mean
            gt = test_Y[ind, i] * std + mean
            delta = np.linalg.norm(pred.flatten() - gt.flatten())
            losses.setdefault(offset, []).append(delta)

    return sorted((k, np.mean(v)) for k, v in losses.items())


def prepare_mocap_data(filename, seq_length):
    poses = np.loadtxt(filename, delimiter=',')
    assert poses.ndim == 2 and poses.shape[1] == 99, poses.shape

    # Take out zero features
    zero_inds, = np.nonzero((poses != 0).any(axis=0))
    assert (zero_inds == GOOD_MOCAP_INDS).all(), zero_inds
    poses = poses[:, GOOD_MOCAP_INDS]

    # Allow sequences to overlap. This substantially increases the amount of
    # training data when sequences are long.
    seqs = []
    end = len(poses) - seq_length + 1
    step = min(seq_length, 50)
    for start in range(0, end, step):
        seqs.append(poses[start:start + seq_length])

    data = np.stack(seqs)
    X = data[:, :-1, :]
    Y = data[:, 1:, :]

    # ERD paper claims to be standardising inputs, but not outputs (?)
    # I standardise both anyway.
    mean = poses.mean(axis=0).reshape((1, 1, -1))
    std = poses.std(axis=0).reshape((1, 1, -1))
    X = (X - mean) / std
    Y = (Y - mean) / std

    # This is sometimes useful for testing
    # full_seq = poses.reshape((1, ) + poses.shape)
    # full_seq = (full_seq - mean) / std

    all_means = np.repeat(mean, X.shape[0], axis=0)
    all_stds = np.repeat(std, X.shape[0], axis=0)

    return X, Y, all_means, all_stds


def load_mocap_data(seq_length):
    fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')

    filenames = glob('h36m-3d-poses/expmap_*.txt.gz')

    train_mean_blocks = []
    test_mean_blocks = []
    train_std_blocks = []
    test_std_blocks = []
    train_X_blocks = []
    train_Y_blocks = []
    test_X_blocks = []
    test_Y_blocks = []
    # test_seqs = []

    for filename in filenames:
        base = path.basename(filename)
        meta = fnre.match(base).groupdict()
        subj_id = int(meta['subject'])

        X, Y, means, stds = prepare_mocap_data(filename, seq_length)

        if subj_id == 5:
            # subject 5 is for testing
            test_X_blocks.append(X)
            test_Y_blocks.append(Y)
            test_mean_blocks.append(means)
            test_std_blocks.append(stds)
            # test_seqs.append(full_seq)
        else:
            train_X_blocks.append(X)
            train_Y_blocks.append(Y)
            train_mean_blocks.append(means)
            train_std_blocks.append(stds)

    train_X = np.concatenate(train_X_blocks, axis=0)
    train_Y = np.concatenate(train_Y_blocks, axis=0)
    train_means = np.concatenate(train_mean_blocks, axis=0)
    train_stds = np.concatenate(train_std_blocks, axis=0)

    test_X = np.concatenate(test_X_blocks, axis=0)
    test_Y = np.concatenate(test_Y_blocks, axis=0)
    test_means = np.concatenate(test_mean_blocks, axis=0)
    test_stds = np.concatenate(test_std_blocks, axis=0)

    return train_X, train_Y, train_means, train_stds, \
        test_X,  test_Y,  test_means,  test_stds


def custom(thing):
    """Decorator to register a custom object"""
    name = thing.__name__
    assert name not in CUSTOM_OBJECTS, "Duplicate object %s" % name
    CUSTOM_OBJECTS[name] = thing
    return thing


def convert_2d_seq(seq):
    """Convert a T*2*8 sequence of poses into a representation more amenable
    for learning. Returns T*F array of features."""
    assert seq.ndim == 3 and seq.shape[1:] == (2, 8)
    rv = seq.copy()

    # Begin by standardising data. Should (approximately) center the person,
    # without distorting width and height.
    rv = (rv - rv.mean()) / rv.std()

    # For non-root nodes, use distance-to-parent only
    # Results in root node (and root node only) storing absolute position
    for j, p in enumerate(PA):
        if j != p:
            rv[:, :, j] = seq[:, :, j] - seq[:, :, p]

    # Track velocity with head, instead of absolute position.
    rv[1:, :, 0] = rv[1:, :, 0] - rv[:-1, :, 0]
    rv[0, :, 0] = [0, 0]

    return rv.reshape((rv.shape[0], -1))


def unmap_predictions(seq, n=1):
    """Convert predictions back into actual poses. Assumes that original
    sequence (including input) was processed with `convert_2d_seq`. Always puts
    position of first predicted head at [0, 0]. Subsequent poses have head
    positions defined by previous ones.

    Returns an array which is (no. samples)*(no. pred offsets)*2*8."""
    seq = seq.reshape((-1, n, 2, 8))
    rv = np.zeros_like(seq)

    # Undo offset-based nonsense for everything below the head (puts the head
    # at zero)
    for joint in range(1, 8):
        parent = PA[joint]
        rv[:, :, :, joint] = seq[:, :, :, joint] + rv[:, :, :, parent]

    # Undo head motion (assumes head at zero in final frame)
    rv[:, :, :, 0] = seq[:, :, :, 0]
    for time in range(1, n):
        delta = seq[:, time, :, 0:1]
        offset = delta + rv[:, time - 1, :, 0:1]
        rv[:, time, :, :] += offset

    return rv


def heatmapify(pose, size, std):
    """Convert a pose to a rows*cols*joints heatmap. Size is rows * cols."""
    assert pose.ndim == 2 and pose.shape[0] == 2
    njoints = pose.shape[1]
    rows, cols = size
    rv = np.zeros((rows, cols, njoints))
    row_x = np.arange(rows)
    col_x = np.arange(cols)
    for j in range(njoints):
        joint_col, joint_row = pose[:, j]
        row_dist = norm.pdf(row_x, loc=joint_row, scale=std)
        col_dist = norm.pdf(col_x, loc=joint_col, scale=std)
        rv[:, :, j] = np.outer(row_dist, col_dist)
    return rv


def unheatmapify(heatmap):
    """Convert a heatmap back to a set of coordinates (taking the mode)"""
    assert heatmap.ndim == 3
    njoints = heatmap.shape[2]
    rv = np.zeros((2, njoints))
    # ravel inner dimensions of heatmap, then recover coordinates for each
    shaped = heatmap.reshape((-1, njoints))
    rows, cols = np.unravel_index(shaped.argmax(axis=0), heatmap.shape[:2])
    # we do [cols, rows] because rv[0] is meant to be x coords (cols) and rv[1]
    # is meant to be y coords (rows)
    rv = np.vstack([cols, rows])
    assert rv.shape == (2, njoints)
    return rv


def heatmapify_batch(poses, size, std=1.5):
    assert poses.ndim == 4 and poses.shape[2] == 2, poses.shape
    B, T, _, J = poses.shape
    rows, cols = size
    rv = np.zeros((B, T, rows, cols, J), dtype='float32')
    for b in range(B):
        for t in range(T):
            pose = poses[b, t]
            rv[b, t] = heatmapify(pose, size, std)
    return rv


def pck(y_true, y_pred, threshs, offsets):
    """Calculate head-neck normalised PCK."""

    nt = len(offsets)
    rv = OrderedDict()

    assert y_true.ndim == 4
    assert y_true.shape == y_pred.shape
    assert y_true.shape[1:] == (nt, 2, 8)

    for thresh in threshs:
        dists = np.linalg.norm(y_true - y_pred, axis=2)
        # Normalise by head-chin distance (or something)
        heads = np.linalg.norm(y_true[:, :, :, 0] - y_true[:, :, :, 1], axis=2)
        pck = (dists < heads.reshape((heads.shape[0], nt, 1)) * thresh) \
            .mean(axis=2) \
            .mean(axis=0)
        for off_i, off in enumerate(offsets):
            label = 'pckh@%.2f/%d' % (thresh, off)
            rv[label] = pck[off_i]

    return rv


@custom
def huber_loss(y_true, y_pred):
    diff = y_true - y_pred
    diff_abs = K.abs(diff)
    error_abs = diff_abs - 0.5
    error_sq = (diff**2) / 2.0
    merged = K.switch(diff_abs <= 1.0, error_sq, error_abs)
    return K.mean(merged)


@custom
class VariableGaussianNoise(Layer):
    """Variant of GaussianNoise that you can turn up or down."""

    def __init__(self, sigma, **kwargs):
        self.supports_masking = True
        self.sigma = K.variable(sigma)
        self.uses_learning_phase = True
        super(VariableGaussianNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):
        noise_x = x + K.random_normal(
            shape=K.shape(x), mean=0., std=self.sigma)
        return K.in_train_phase(noise_x, x)

    def get_sigma(self):
        return K.get_value(self.sigma)

    def set_sigma(self, new_sigma):
        K.set_value(self.sigma, new_sigma)

    def get_config(self):
        config = {'sigma': K.get_value(self.sigma)}
        base_config = super(VariableGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@custom
class VariableScaler(Layer):
    """Constant scaler with a variable coefficient."""

    def __init__(self, scale, **kwargs):
        self.scale = K.variable(scale)
        super(VariableScaler, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x * self.scale

    def get_scale(self):
        return K.get_value(self.scale)

    def set_scale(self, new_scale):
        K.set_value(self.scale, new_scale)

    def get_config(self):
        config = {'scale': K.get_value(self.scale)}
        base_config = super(VariableScaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PatienceCallback(Callback):
    """Base class for things which wait for loss to plateau"""

    def __init__(self, patience, quantity='val_loss'):
        self.waiting = 0
        self.best_loss = float('inf')
        self.quantity = quantity
        self.should_update = True
        self.patience = patience

    def on_epoch_begin(self, epoch, logs={}):
        """Do update at beginning of epoch instead of the end. This
        ensures that we always update during the first epoch."""
        if self.should_update:
            self.do_update()
            self.should_update = False

    def on_epoch_end(self, epoch, logs):
        epoch_loss = logs[self.quantity]
        improved = epoch_loss < self.best_loss
        expired = self.waiting > self.patience
        if expired or improved:
            # If we've improved or run out of time, reset counters
            self.best_loss = epoch_loss if improved else float('inf')
            self.waiting = 0
        if expired and not improved:
            self.should_update = True
        self.waiting += 1


class GaussianRamper(PatienceCallback):
    """Ramps up magnitude of Gaussian noise as model converges."""

    def __init__(self, patience, schedule, **kwargs):
        self.sched_iter = iter(schedule)
        super(GaussianRamper, self).__init__(patience, **kwargs)

    def get_noise_layers(self):
        for layer in self.model.layers:
            if isinstance(layer, VariableGaussianNoise):
                yield layer

    def do_update(self):
        try:
            next_noise = next(self.sched_iter)
        except StopIteration:
            print('Gaussian ramper out of patience, but schedule exhausted')
            return
        n = 0
        for layer in self.get_noise_layers():
            layer.set_sigma(next_noise)
            n += 1
        print('Ramping Gaussian noise up to %g on %d layers' % (next_noise, n))


class ScaleRamper(PatienceCallback):
    """Like GaussianRamper, but ramps up the scale of some layer. Currently
    (Feb 2017) used to ramp up KL divergence on VAEs"""

    def __init__(self, patience, schedule, target_name, **kwargs):
        self.sched_iter = iter(schedule)
        # may want to change to multiple target names later (hence set)
        self.target_names = {target_name}
        super(ScaleRamper, self).__init__(patience, **kwargs)

    def get_target_layers(self):
        for layer in self.model.layers:
            if isinstance(layer, VariableScaler) \
               and layer.name in self.target_names:
                yield layer

    def do_update(self):
        try:
            next_scale = next(self.sched_iter)
        except StopIteration:
            print('Scale ramper out of patience, but schedule exhausted')
            return
        n = 0
        for layer in self.get_target_layers():
            layer.set_scale(next_scale)
            n += 1
        print('Ramping scale noise up to %g on %d layers' % (next_scale, n))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken from sklearn examples:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
