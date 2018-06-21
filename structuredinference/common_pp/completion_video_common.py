"""Common (dataset-shared) code for making completino videos"""

import re
import os

import numpy as np

from scipy.optimize import fmin

_num_re = re.compile(r'(\d+)')


def load_sorted_paths(frame_dir):
    """Sorts a bunch of paths that have numbers in the filename."""
    fns = os.listdir(frame_dir)
    everything = []
    for fn in fns:
        bn = os.path.basename(fn)
        # if we get many numbers then there is a bug
        num_str, = _num_re.findall(bn)
        thing_id = int(num_str)
        everything.append((thing_id, os.path.join(frame_dir, fn)))
    return [p for i, p in sorted(everything)]


def alignment_constant(rec_x, true_x):
    # This almost certainly has a simple analytic solution, but I can't be
    # bothered finding it right now. Instead, I'm centring both, scaling until
    # they match, then returning alpha and beta required to do the scaling
    # for other samples.
    # Update: ...unsurprisingly, this doesn't work very well :(
    # expect single poses (2*j)
    assert true_x.shape == rec_x.shape
    assert true_x.shape[0] == 2 and rec_x.shape[0] == 2
    assert true_x.ndim == 2 and rec_x.ndim == 2
    rec_cen = rec_x - rec_x.mean(axis=1)[:, None]
    true_cen = true_x - true_x.mean(axis=1)[:, None]

    def objective(a):
        return np.sqrt(np.sum((rec_cen * a - true_cen).flatten()**2))

    opt_result = fmin(objective, x0=19)
    alpha, = opt_result
    # to reconstruct: (rec_x - rec_x.mean(axis=1)) * alpha +
    # true_x.mean(axis=1)
    beta = true_x.mean(axis=1) - alpha * rec_x.mean(axis=1)
    return alpha, beta
