"""Utilities for loading 2D pose sequences"""

import numpy as np
from scipy import stats, signal


def gauss_filter(seq, sigma, filter_width=None):
    if filter_width is None:
        filter_width = int(np.ceil(4 * sigma))
        if (filter_width % 2) == 0:
            # make it odd-length (not sure how things work with even-length
            # filters, TBH)
            filter_width += 1
    assert filter_width > 0, 'need a reasonably large filter'
    extent = filter_width / 2.0
    x_vals = np.linspace(-extent, extent, num=filter_width)
    kern = stats.norm.pdf(x_vals, loc=0, scale=sigma)
    kern = kern / np.sum(kern)
    kern = kern.reshape((-1, 1))
    # a 2D convolution is not strictly necessary (this is really only a 1D
    # convolution across each channel), but I'm using it anyway because it
    # supports boundary='symm' (symmetric padding)
    smoothed = signal.convolve2d(seq, kern, mode='same', boundary='symm')
    assert smoothed.shape == seq.shape
    return smoothed


def preprocess_sequence(poses, parents, smooth=False):
    """Preprocess a sequence of 2D poses to have more tractable representation.
    `parents` array is used to calculate output entries which are
    parent-relative joint locations. Note that standardisation will have to be
    performed later."""
    # Poses should be T*(XY)*J
    assert poses.ndim == 3, poses.shape
    assert poses.shape[1] == 2, poses.shape

    if smooth:
        pflat = poses.reshape((poses.shape[0], np.prod(poses.shape[1:])))
        # remove high freq noise with reasonably narrow Gaussian
        psmooth = gauss_filter(pflat, sigma=1)
        poses = psmooth.reshape(poses.shape)

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


def reconstruct_poses(flat_poses, parents):
    """Undo parent-relative joint transform. Will not undo the uniform scaling
    applied to each sequence."""
    # shape of poses shold be (num training samples)*(time)*(flattened
    # dimensions)
    assert flat_poses.ndim == 3, flat_poses.shape

    # rel_poses is a 4D array: (num samples)*T*(XY)*J
    rel_poses = flat_poses.reshape(flat_poses.shape[:2] + (2, -1))
    true_poses = np.zeros_like(rel_poses)
    N, T, Dxy, J = true_poses.shape
    assert Dxy == 2
    assert len(parents) == J

    # start by restoring head from velocity param
    rel_heads = rel_poses[:, :, :, 0]
    true_heads = np.cumsum(rel_heads, axis=1)
    true_poses[:, :, :, 0] = true_heads

    # now reconstruct remaining joints from parents
    for joint in range(1, len(parents)):
        parent = parents[joint]
        parent_pos = true_poses[:, :, :, parent]
        offsets = rel_poses[:, :, :, joint]
        true_poses[:, :, :, joint] = parent_pos + offsets

    return true_poses


def runs(vec):
    assert vec.ndim == 1
    # mask vec indicating whether given element is a run end
    run_ends = np.empty_like(vec, dtype='bool')
    run_ends[-1] = True
    run_ends[:-1] = vec[:-1] != vec[1:]
    run_length = np.cumsum(run_ends)

    run_stops = np.nonzero(run_ends) + 1
    run_starts = np.empty_like(vec, dtype='int')
    run_starts[0] = 0
    run_starts[1:] = run_end_inds[:-1]

    act_vals = vec[run_starts]

    return list(zip(act_vals, starts, stops))


def extract_action_dataset(feats, actions, min_length=10):
    """Given pose sequence and action, return pairs of form (pose sequence, action label)"""
    # need T*D features (T time, D dimensionality of features)
    assert feats.ndim == 2, poses.shape
    # actions should be single array of action numbers
    assert actions.ndim == 1, actions.shape
    pairs = []
    for action, start, stop in runs(actions):
        if stop - start < min_length:
            continue
        pairs.append((feats[start:stop], action))
    return pairs
