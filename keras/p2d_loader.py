"""Utilities for loading 2D pose sequences"""

import numpy as np
from scipy import stats, signal
import json
import h5py


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


def preprocess_sequence(poses, parents, smooth_sigma=False, remove_head=False):
    """Preprocess a sequence of 2D poses to have more tractable representation.
    `parents` array is used to calculate output entries which are
    parent-relative joint locations. Note that standardisation will have to be
    performed later."""
    # Poses should be T*(XY)*J
    assert poses.ndim == 3, poses.shape
    assert poses.shape[1] == 2, poses.shape

    if smooth_sigma:
        # Remove high freq noise with reasonably narrow Gaussian. Channel-wise,
        # so can be applied before anything else.
        pflat = poses.reshape((poses.shape[0], np.prod(poses.shape[1:])))
        psmooth = gauss_filter(pflat, sigma=smooth_sigma)
        poses = psmooth.reshape(poses.shape)

    if remove_head:
        poses, parents = remove_head(poses, parents)

    # Scale so that person roughly fits in 1x1 box at origin
    scale = (np.max(poses, axis=2) - np.min(poses, axis=2)).flatten().std()
    assert 1e-3 < scale < 1e4, scale
    offset = np.mean(np.mean(poses, axis=2), axis=0).reshape((1, 2, 1))
    norm_poses = (poses - offset) / scale

    if parents is not None:
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
    else:
        shaped = poses.reshape((norm_poses.shape[0], -1))

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

    for start, stop, act in zip(run_starts, run_stops, act_vals):
        assert np.all(vec[start:stop] == act)

    return list(zip(act_vals, run_starts, run_stops))


def extract_action_dataset(feats, actions, seq_length, gap):
    """Given pose sequence and action, return pairs of form (pose sequence,
    action label)"""
    # need T*D features (T time, D dimensionality of features)
    assert feats.ndim == 2, feats.shape
    # actions should be single array of action numbers
    assert actions.ndim == 1, actions.shape
    pairs = []
    subseqs = 0
    too_short = 0
    all_runs = runs(actions)
    for action, start, stop in all_runs:
        length = stop - start
        if length < seq_length:
            too_short += 1
            continue
        for sub_start in range(start, stop - seq_length + 1, gap):
            # no need to temporally downsample; features have already been
            # temporally downsampled
            pairs.append((feats[sub_start:sub_start + seq_length], action))
            subseqs += 1
    return pairs


def load_p2d_data(data_file,
                  seq_length,
                  seq_skip,
                  gap=1,
                  val_frac=0.2,
                  add_noise=0.6,
                  load_actions=True,
                  completion_length=None,
                  relative=True):
    """Preprocess an open HDF5 file which has been formatted to hold 2D poses.

    :param data_file: The open HDF5 file.
    :param seq_length: Length of output sequences.
    :param seq_skip: How many (original) frames apart each pose should be in an
        output sequence.
    :param gap: How far to skip forward between each sampled sequence.
    :param val_frac: Percentage of dataset to use as validation data.
    :param add_noise: Change one-hot action vectors to be distributions which
        select correct action ``add_noise*100``% of the time (or choose
        randomly otherwise). Hack to emulate noisy actions.
    :param load_actions: Should actions actually be returned?
    :param completino_length: Number of sequential poses to use in completion
        problems. Set to None to disable.
    :param relative: Whether to use parent-relative parameterisation.
    :rtype: Dictionary of relevant data."""
    train_pose_blocks = []
    val_pose_blocks = []
    train_mask_blocks = []
    val_mask_blocks = []
    if load_actions:
        train_action_blocks = []
        train_aclass_ds = []
        val_action_blocks = []
        val_aclass_ds = []
        # gap is smaller for action classification sequences because we have
        # already downsampled
        # aclass_gap = max(1, int(np.ceil(gap / float(seq_skip))))
        # aclass_gap = gap
        aclass_gap = seq_length
    else:
        train_aclass_ds = val_aclass_ds = None
    if completion_length:
        train_completions = val_completions = None

    # for deterministic val set split
    srng = np.random.RandomState(seed=8904511)

    with h5py.File(data_file, 'r') as fp:
        parents = fp['/parents'].value

        if load_actions:
            num_actions = fp['/num_actions'].value.flatten()[0]
            action_json_string = fp['/action_names'].value.tostring().decode(
                'utf8')
            action_names = ['n/a'] + json.loads(action_json_string)
        else:
            action_names = None

        vid_names = list(fp['seqs'])
        val_vid_list = list(vid_names)
        srng.shuffle(val_vid_list)
        val_count = max(1, int(val_frac * len(val_vid_list)))
        val_vids = set(val_vid_list[:val_count])
        train_vids = set(val_vid_list) - val_vids

        missing_mask = False

        for vid_name in fp['seqs']:
            poses = fp['/seqs/' + vid_name + '/poses'].value
            # don't both with relative poses
            norm_poses = preprocess_sequence(
                poses, parents if relative else None, smooth_sigma=2)

            if load_actions:
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
                assert len(norm_poses) == len(one_hot_acts)

                aclass_list = extract_action_dataset(norm_poses, actions,
                                                     seq_length, aclass_gap)
                if vid_name in val_vids:
                    val_aclass_ds.extend(aclass_list)
                else:
                    train_aclass_ds.extend(aclass_list)

            if '/seqs/' + vid_name + '/valid' in fp:
                mask = fp['/seqs/' + vid_name + '/valid'].value
                mask = mask.reshape((mask.shape[0], -1))
                assert mask.shape == norm_poses.shape, \
                    "mask should be %s, but was %s" % (norm_poses.shape,
                                                       mask.shape)
            else:
                missing_mask = True
                mask = np.ones_like(norm_poses)

            range_count = len(norm_poses) - seq_skip * seq_length + 1
            for i in range(0, range_count, gap):
                pose_block = norm_poses[i:i + seq_skip * seq_length:seq_skip]
                mask_block = mask[i:i + seq_skip * seq_length:seq_skip]

                if load_actions:
                    sk = seq_skip
                    sl = seq_length
                    act_block = one_hot_acts[i:i + sk * sl:sk]

                if vid_name in val_vids:
                    train_pose_blocks.append(pose_block)
                    if load_actions:
                        train_action_blocks.append(act_block)
                    train_mask_blocks.append(mask_block)
                else:
                    val_pose_blocks.append(pose_block)
                    if load_actions:
                        val_action_blocks.append(act_block)
                    val_mask_blocks.append(mask_block)

            if completion_length:
                # choose non-overlapping seqeuences for completion dataset
                range_bound = 1 + len(norm_poses) \
                              - seq_skip * completion_length + 1
                for i in range(0, range_bound, seq_skip):
                    endpoint = i + seq_skip * completion_length
                    pose_block = norm_poses[i:endpoint:seq_skip]
                    mask_block = mask[i:endpoint:seq_skip]
                    assert pose_block.shape == mask_block.shape, \
                        'poses %s, mask %s' % (pose_block.shape,
                                               mask_block.shape)
                    assert len(pose_block) == completion_length, \
                        pose_block.shape
                    completion_block = {
                        'poses': pose_block,
                        'mask': mask_block,
                        'vid_name': vid_name,
                        'start': i,
                        'stop': endpoint,
                        'skip': seq_skip
                    }
                    if vid_name in val_vids:
                        val_completions.append(completion_block)
                    else:
                        train_completions.append(completion_block)

    if missing_mask:
        print('Some masks found missing by loader; assuming unmasked')
    else:
        print('Loader using true masks from file')

    train_poses = np.stack(train_pose_blocks, axis=0).astype('float32')
    train_mask = np.stack(train_mask_blocks, axis=0).astype('float32')
    val_poses = np.stack(val_pose_blocks, axis=0).astype('float32')
    val_mask = np.stack(val_mask_blocks, axis=0).astype('float32')
    if load_actions:
        train_actions = np.stack(train_action_blocks, axis=0).astype('float32')
        val_actions = np.stack(val_action_blocks, axis=0).astype('float32')
    else:
        train_actions = val_actions = None

    flat_poses = train_poses.reshape((-1, train_poses.shape[-1]))
    mean = flat_poses.mean(axis=0).reshape((1, 1, -1))
    std = flat_poses.std(axis=0).reshape((1, 1, -1))
    # setting low stds to 1 will have effect of making low-variance features
    # (almost) constant zero
    std[std < 1e-5] = 1
    train_poses = (train_poses - mean) / std
    val_poses = (val_poses - mean) / std
    train_poses[train_mask == 0] = 0
    val_poses[val_mask == 0] = 0
    for action_ds in [train_aclass_ds, val_aclass_ds]:
        for f, _ in action_ds:
            # TODO: what else to do here? Should I save masks and remove those?
            f[:] = (f - mean) / std
    for completion_ds in [train_completions, val_completions]:
        for comp_dict in completion_ds:
            poses = comp_dict['poses']
            f[:] = (f - mean) / std
            f[comp_dict['mask'] == 0] = 0

    return {
        'train_poses': train_poses,
        'train_mask': train_mask,
        'train_actions': train_actions,
        'val_poses': val_poses,
        'val_mask': val_mask,
        'val_actions': val_actions,
        'mean': mean,
        'std': std,
        'action_names': action_names,
        'train_vids': sorted(train_vids),
        'val_vids': sorted(val_vids),
        'data_path': data_file,
        'parents': parents,
        'train_aclass_ds': train_aclass_ds,
        'val_aclass_ds': val_aclass_ds,
        'train_completions': train_completions,
        'val_completions': val_completions
    }
