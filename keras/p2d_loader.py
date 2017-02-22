"""Utilities for loading 2D pose sequences"""

import numpy as np
from scipy import stats, signal
import json
import h5py
from collections import namedtuple


def gauss_filter(seq, sigma, filter_width=None):
    """Filter a 2D signal, where first dimension is time, and second is data
    channels."""
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


def is_toposorted_tree(parents):
    """Check that parents array defines toposorted tree (root-first)."""
    root_connected = [False] * len(parents)
    for ch, pa in enumerate(parents):
        if ch == 0:
            # root must loop back on itself
            invalid = pa != 0
        else:
            # bad scenarios: parent might occur later, or might not be
            # connected to a root (and thus form a separate tree in a forest)
            invalid = pa >= ch or not root_connected[pa]
        if invalid:
            return False
        root_connected[ch] = True
    return True


def remove_head(poses, parents):
    """Removes joint 0 (usually head) from the pose tree. At the moment, it
    requires joint 0 to have only one child (but there's no theoretical reason
    for that; you could just as easily average several children to create a new
    root). Could also extend to removing arbitrary joints relatively easily."""
    assert is_toposorted_tree(parents)

    children = [ch for ch, pa in enumerate(parents) if pa == 0 and ch != 0]
    assert children[0] <= 1, "only supports one child at the moment"

    new_parents = [p - 1 for p in parents[1:]]
    assert is_toposorted_tree(new_parents)
    assert poses.ndim == 3, "need poses (shape %s) to be T,XY,J" % poses.shape
    new_poses = poses[:, :, 1:]

    return new_poses, new_parents


def preprocess_sequence(poses, parents, smooth=False, remove_head=False):
    """Preprocess a sequence of 2D poses to have more tractable representation.
    `parents` array is used to calculate output entries which are
    parent-relative joint locations. Note that standardisation will have to be
    performed later."""
    # Poses should be T*(XY)*J
    assert poses.ndim == 3, poses.shape
    assert poses.shape[1] == 2, poses.shape

    if smooth:
        # Remove high freq noise with reasonably narrow Gaussian. Channel-wise,
        # so can be applied before anything else.
        pflat = poses.reshape((poses.shape[0], np.prod(poses.shape[1:])))
        psmooth = gauss_filter(pflat, sigma=1)
        poses = psmooth.reshape(poses.shape)

    if remove_head:
        poses, parents = remove_head(poses, parents)

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
    vec = np.asarray(vec)
    assert vec.ndim == 1

    # mask vec indicating whether given element is a run end
    run_ends = np.empty_like(vec, dtype='bool')
    run_ends[-1] = True
    run_ends[:-1] = vec[:-1] != vec[1:]

    # Now we get indices. start:stop is meant to delimit values.
    run_stops = np.nonzero(run_ends)[0] + 1
    run_starts = np.empty_like(run_stops, dtype='int')
    run_starts[0] = 0
    run_starts[1:] = run_stops[:-1]

    act_vals = vec[run_starts]

    return list(zip(act_vals, run_starts, run_stops))


def extract_action_dataset(feats, actions, min_length=5):
    """Given pose sequence and action, return pairs of form (pose sequence,
    action label)"""
    # need T*D features (T time, D dimensionality of features)
    assert feats.ndim == 2, feats.shape
    # actions should be single array of action numbers
    assert actions.ndim == 1, actions.shape
    pairs = []
    total_len = 0
    all_runs = runs(actions)
    for action, start, stop in all_runs:
        length = stop - start
        if length < min_length:
            continue
        total_len += length
        pairs.append((feats[start:stop], action))
    if len(pairs) < len(all_runs):
        print('%i/%i sequences were too short; still got %i frames, though' %
              (len(all_runs) - len(pairs), len(all_runs), total_len))
    return pairs


Data = namedtuple('Data', [
    'train_poses', 'train_actions', 'val_poses', 'val_actions', 'mean', 'std',
    'action_names', 'train_vids', 'val_vids', 'data_path', 'parents',
    'train_aclass_ds', 'val_aclass_ds'
])


def load_p2d_data(data_file,
                  seq_length,
                  seq_skip,
                  gap=1,
                  val_frac=0.2,
                  add_noise=0.6):
    train_pose_blocks = []
    train_action_blocks = []
    train_aclass_ds = []
    val_pose_blocks = []
    val_action_blocks = []
    val_aclass_ds = []

    # for deterministic val set split
    srng = np.random.RandomState(seed=8904511)

    with h5py.File(data_file, 'r') as fp:
        parents = fp['/parents'].value
        num_actions = fp['/num_actions'].value.flatten()[0]

        action_json_string = fp['/action_names'].value.tostring().decode(
            'utf8')
        action_names = ['n/a'] + json.loads(action_json_string)

        vid_names = list(fp['seqs'])
        val_vid_list = list(vid_names)
        srng.shuffle(val_vid_list)
        val_count = max(1, int(val_frac * len(val_vid_list)))
        val_vids = set(val_vid_list[:val_count])
        train_vids = set(val_vid_list) - val_vids

        for vid_name in fp['seqs']:
            actions = fp['/seqs/' + vid_name + '/actions'].value
            # `cert` chance of choosing correct action directly, `1 - cert`
            # chance of choosing randomly (maybe gets correct action)
            if add_noise is not None:
                cert = add_noise
            else:
                cert = 1
            one_hot_acts = (1 - cert) * np.ones(
                (len(actions), num_actions + 1)) / (num_actions + 1)
            # XXX: This is an extremely hacky way of injecting noise :/
            one_hot_acts[(range(len(actions)), actions)] += cert
            # actions should form prob dist., roughly
            assert np.all(np.abs(1 - one_hot_acts.sum(axis=1)) < 0.001)

            poses = fp['/seqs/' + vid_name + '/poses'].value
            relposes = preprocess_sequence(poses, parents, smooth=True)

            assert len(relposes) == len(one_hot_acts)

            aclass_list = extract_action_dataset(relposes, actions)
            if vid_name in val_vids:
                val_aclass_ds.extend(aclass_list)
            else:
                train_aclass_ds.extend(aclass_list)

            for i in range(len(relposes) - seq_skip * seq_length + 1):
                pose_block = relposes[i:i + seq_skip * seq_length:seq_skip]
                act_block = one_hot_acts[i:i + seq_skip * seq_length:seq_skip]

                if vid_name in val_vids:
                    train_pose_blocks.append(pose_block)
                    train_action_blocks.append(act_block)
                else:
                    val_pose_blocks.append(pose_block)
                    val_action_blocks.append(act_block)

    train_poses = np.stack(train_pose_blocks, axis=0).astype('float32')
    train_actions = np.stack(train_action_blocks, axis=0).astype('float32')
    val_poses = np.stack(val_pose_blocks, axis=0).astype('float32')
    val_actions = np.stack(val_action_blocks, axis=0).astype('float32')

    flat_poses = train_poses.reshape((-1, train_poses.shape[-1]))
    mean = flat_poses.mean(axis=0).reshape((1, 1, -1))
    std = flat_poses.std(axis=0).reshape((1, 1, -1))
    # TODO: Smarter handling of std. Will also need to use smarter
    # handling in actual loader script used by train.py
    std[std < 1e-5] = 1
    train_poses = (train_poses - mean) / std
    val_poses = (val_poses - mean) / std

    return Data(train_poses, train_actions, val_poses, val_actions, mean, std,
                action_names,
                sorted(train_vids),
                sorted(val_vids), data_file, parents, train_aclass_ds,
                val_aclass_ds)
