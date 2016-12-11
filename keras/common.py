"""Code shared between recurrent and non-recurrent pose prediction models."""

from collections import OrderedDict

from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Layer
import numpy as np
from scipy.stats import norm

PA = [0, 0, 1, 2, 3, 1, 5, 6]
THRESHOLDS = [0.05, 0.15, 0.25, 0.5, 0.8]
# Dictionary to pass to load_model, etc., to find custom objectives and layers
CUSTOM_OBJECTS = {}


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


def _ax_hmfy(ax_loc, length, std):
    # Trying to reshape so that we get an output which is (num joints * length)
    x = np.arange(length)
    y = norm.pdf(x, loc=ax_loc, scale=std)
    assert ax_locs.ndim == 1
    broad_locs = ax_locs.reshape((-1, 1))
    norm.pdf(x)


def heatmapify(pose, size, std=1.5):
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
    error_sq = (diff ** 2) / 2.0
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
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      std=self.sigma)
        return K.in_train_phase(noise_x, x)

    def get_sigma(self):
        return K.get_value(self.sigma)

    def set_sigma(self, new_sigma):
        K.set_value(self.sigma, new_sigma)

    def get_config(self):
        config = {'sigma': K.get_value(self.sigma)}
        base_config = super(VariableGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianRamper(Callback):
    """Ramps up magnitude of Gaussian noise as model converges."""
    def __init__(self, patience, schedule, quantity='val_loss'):
        self.sched_iter = iter(schedule)
        self.waiting = 0
        self.best_loss = float('inf')
        self.quantity = quantity
        self.update_sigma = True
        self.patience = patience

    def get_noise_layers(self):
        for layer in self.model.layers:
            if isinstance(layer, VariableGaussianNoise):
                yield layer

    def on_epoch_begin(self, epoch, logs={}):
        """Do noise update at beginning of epoch instead of the end. This
        ensures that we always update during the first epoch."""
        if self.update_sigma:
            self.update_sigma = False
            try:
                next_noise = next(self.sched_iter)
            except StopIteration:
                print('Ramper out of patience, but schedule exhausted')
                return
            n = 0
            for layer in self.get_noise_layers():
                layer.set_sigma(next_noise)
                n += 1
            print('Ramping Gaussian noise up to %g on %d layers' % (next_noise, n))

    def on_epoch_end(self, epoch, logs):
        epoch_loss = logs[self.quantity]
        improved = epoch_loss < self.best_loss
        expired = self.waiting > self.patience
        if expired or improved:
            # If we've improved or run out of time, reset counters
            self.best_loss = epoch_loss if improved else float('inf')
            self.waiting = 0
        if expired and not improved:
            self.update_sigma = True
        self.waiting += 1
