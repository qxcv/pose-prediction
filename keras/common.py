"""Code shared between recurrent and non-recurrent pose prediction models."""

from collections import OrderedDict

from keras import backend as K
from keras import objectives, metrics
import numpy as np

PA = [0, 0, 1, 2, 3, 1, 5, 6]
THRESHOLDS = [0.05, 0.15, 0.25, 0.5, 0.8]


def huber_loss(y_true, y_pred):
    diff = y_true - y_pred
    diff_abs = K.abs(diff)
    error_abs = diff_abs - 0.5
    error_sq = (diff ** 2) / 2.0
    merged = K.switch(diff_abs <= 1.0, error_sq, error_abs)
    return K.mean(merged)
# This helps Keras model loader find huber_loss
objectives.huber_loss = huber_loss
metrics.huber_loss = huber_loss


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
