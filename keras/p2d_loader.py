"""Utilities for loading 2D pose sequences"""

import numpy as np
from scipy import stats, signal
import json
import h5py
import pandas as pd


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


def head_remover(parents):
    """Removes joint 0 (usually head) from the pose tree. At the moment, it
    requires joint 0 to have only one child (but there's no theoretical reason
    for that; you could just as easily average several children to create a new
    root). Could also extend to removing arbitrary joints relatively easily.

    Returns the parents array and a transformation matrix to convert poses."""
    assert is_toposorted_tree(parents)

    # TODO: consider support for removing several joints. Will require change
    # in transition matrix!
    children = [ch for ch, pa in enumerate(parents) if pa == 0 and ch != 0]
    assert children[0] <= 1, "only supports one child at the moment"

    new_parents = [p - 1 if p != 0 else 0 for p in parents[1:]]
    assert is_toposorted_tree(new_parents)

    transition = np.concatenate(
        (np.zeros((len(new_parents), 1)), np.eye(len(new_parents))), axis=1)
    assert transition.shape == (len(new_parents), len(parents))

    return transition, new_parents


def preprocess_sequence(poses, skip, parents, smooth_sigma=False,
                        hrm_mat=None):
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

    if hrm_mat is not None:
        assert hrm_mat.ndim == 2
        # poses is T*(XY)*J, and hrm_mat is J'*J. We want to get back a
        # T*(XY)*J' array (i.e. go down the vectors in the last dimension and
        # multiply by them).
        removed_poses = np.einsum('kl,ijl->ijk', hrm_mat, poses)
        assert removed_poses.shape == poses.shape[:2] + (len(parents), )
        poses = removed_poses

    # Scale so that person roughly fits in 1x1 box at origin
    scale = (np.max(poses, axis=2) - np.min(poses, axis=2)).flatten().std()
    assert 1e-3 < scale < 1e4, scale
    offset = np.mean(np.mean(poses, axis=2), axis=0).reshape((1, 2, 1))
    norm_poses = (poses - offset) / scale

    if parents is not None:
        # Compute actual data (relative offsets are easier to learn)
        relpose = np.zeros_like(norm_poses)
        # Record delta from frame which is <skip> steps in the past
        relpose[skip:, :, 0] \
            = norm_poses[skip:, :, 0] - norm_poses[:-skip, :, 0]
        # Other norm_poses record delta from parents
        for jt in range(1, len(parents)):
            pa = parents[jt]
            relpose[:, :, jt] = norm_poses[:, :, jt] - norm_poses[:, :, pa]

        # Collapse in last two dimensions, interleaving X and Y coordinates
        shaped = relpose.reshape((relpose.shape[0], -1))
    else:
        shaped = poses.reshape((norm_poses.shape[0], -1))

    return shaped, offset, scale


def _reconstruct_poses(flat_poses, parents, pp_offset, pp_scale):
    """Undo parent-relative joint transform. Will not undo the uniform scaling
    applied to each sequence."""
    # shape of poses shold be (num training samples)*(time)*(flattened
    # dimensions)
    assert flat_poses.ndim == 3, flat_poses.shape
    assert pp_offset.size == 2, 'expect offset to be XY vec'
    assert np.asarray(pp_scale).size == 1, 'expect scale to be scalar'

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

    # add back pp_offset and pp_scale
    true_poses = true_poses * pp_scale + pp_offset.reshape((1, 1, 2, 1))

    return true_poses


def _runs(vec):
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


class P2DDataset(object):
    def __init__(self,
                 data_file_path,
                 seq_length,
                 seq_skip,
                 gap=1,
                 val_frac=0.2,
                 have_actions=True,
                 completion_length=None,
                 relative=True,
                 aclass_full_length=128,
                 aclass_act_length=8,
                 remove_head=False):
        """Preprocess an open HDF5 file which has been formatted to hold 2D
        poses.

        :param data_file: The open HDF5 file.
        :param seq_length: Length of output sequences.
        :param seq_skip: How many (original) frames apart each pose should be
            in an output sequence.
        :param gap: How far to skip forward between each sampled sequence.
        :param val_frac: Percentage of dataset to use as validation data.
        :param have_actions: Should actions actually be returned?
        :param completion_length: Number of sequential poses to use in
            completion problems. Set to None to disable.
        :param relative: Whether to use parent-relative parameterisation.
        :param aclass_full_length: length of action classification sequences.
        :param aclass_act_length: how much of sequence actual action must take
            up in action classification sequence. An action classification
            sequence will thus consist of aclass_full_length -
            aclass_act_length poses in sequence (arbitrary actions) followed by
            aclass_act_length poses of the right action.
        :param remove_head: Whether to remove head joint."""
        self.data_file_path = data_file_path
        self.seq_length = seq_length
        self.seq_skip = seq_skip
        self.gap = gap
        self.val_frac = val_frac
        self.have_actions = have_actions
        self.relative = relative
        self.remove_head = remove_head
        self.completion_length = completion_length
        self.aclass_full_length = aclass_full_length
        self.aclass_act_length = 8
        videos_list = []

        with h5py.File(data_file_path, 'r') as fp:
            self.parents = fp['/parents'].value

            if self.remove_head:
                # head removal matrix
                hrm_mat, self.parents = head_remover(self.parents)
            else:
                hrm_mat = None

            if self.have_actions:
                # Redundancy in action count is because I used to write all
                # actions *except the null action* to the action file, and
                # wanted to check that action numbers were correct on this end.
                # Keeping redundancy just to show where brokenness is.
                num_real_actions = fp['/num_actions'].value.flatten()[0]
                action_json_string \
                    = fp['/action_names'].value.tostring().decode('utf8')
                self.action_names = json.loads(action_json_string)
                assert len(self.action_names) == num_real_actions, \
                    'expected %d actions, got %d (incl. n/a)' \
                    % (num_real_actions, len(self.action_names))
                self.num_actions = num_real_actions
            else:
                self.action_names = None

            vid_names = list(fp['seqs'])
            self.train_vids, self.val_vids \
                = self._train_test_split(vid_names, self.val_frac)

            for vid_name in fp['seqs']:
                orig_poses = fp['/seqs/' + vid_name + '/poses'].value
                # don't both with relative poses
                norm_poses, pp_offset, pp_scale = preprocess_sequence(
                    orig_poses,
                    self.seq_skip,
                    self.parents if self.relative else None,
                    smooth_sigma=2,
                    hrm_mat=hrm_mat)
                norm_poses = norm_poses.astype('float32')
                # make sure we don't reuse poses in native parameterisation
                del orig_poses

                if self.have_actions:
                    actions = fp['/seqs/' + vid_name + '/actions'].value
                    assert len(actions) == len(norm_poses)

                if '/seqs/' + vid_name + '/valid' in fp:
                    mask = fp['/seqs/' + vid_name + '/valid'].value
                    mask = mask.reshape((mask.shape[0], -1)).astype('float32')
                    assert mask.shape == norm_poses.shape, \
                        "mask should be %s, but was %s" % (norm_poses.shape,
                                                           mask.shape)
                else:
                    mask = np.ones_like(norm_poses, dtype='float32')

                vid_meta = {
                    'pp_offset': pp_offset,
                    'pp_scale': pp_scale,
                    'poses': norm_poses,
                    'vid_name': vid_name,
                    'mask': mask,
                    'is_val': vid_name in self.val_vids
                }
                if self.have_actions:
                    vid_meta['actions'] = actions

                videos_list.append(vid_meta)

        self.videos = pd.DataFrame.from_records(videos_list)

        train_vids = self.videos[~self.videos.is_val]
        flat_poses = np.concatenate(train_vids.poses.as_matrix(), axis=0)
        assert flat_poses.ndim == 2, flat_poses.shape
        self.mean = flat_poses.mean(axis=0).reshape((1, -1))
        self.std = flat_poses.std(axis=0).reshape((1, -1))
        # setting low std to 1 will have effect of making low-variance features
        # (almost) constant zero
        self.std[self.std < 1e-5] = 1
        for poses, mask in zip(self.videos['poses'], self.videos['mask']):
            poses[:] = (poses[:] - self.mean) / self.std
            # TODO: should I leave this in? Does it make things better?
            poses[mask == 0] = 0

        self.dim_obs = self.mean.size

    @staticmethod
    def _train_test_split(vid_names, val_frac):
        # copy list so we can shuffle
        vid_list = list(vid_names)
        val_srng = np.random.RandomState(seed=8904511)
        val_srng.shuffle(vid_list)
        val_count = max(1, int(val_frac * len(vid_list)))
        val_vids = set(vid_list[:val_count])
        train_vids = set(vid_list) - val_vids
        return train_vids, val_vids

    def get_pose_ds(self, train):
        # poses should be N*T*D, actions should be N*T*A if they exist
        # (one-hot), mask should be N*T*D
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        seq_skip = self.seq_skip
        seq_length = self.seq_length
        gap = self.gap

        pose_blocks = []
        mask_blocks = []
        action_blocks = []

        for vid_idx in vids.index:
            vid_poses = vids['poses'][vid_idx]
            vid_mask = vids['mask'][vid_idx]
            if self.have_actions:
                vid_actions = vids['actions'][vid_idx]
                vid_one_hot_acts = np.zeros((len(vid_actions),
                                             self.num_actions),
                                            dtype='float32')

            range_count = len(vid_poses) - seq_skip * seq_length + 1

            for i in range(0, range_count, gap):
                pose_block = vid_poses[i:i + seq_skip * seq_length:seq_skip]
                pose_blocks.append(pose_block)
                mask_block = vid_mask[i:i + seq_skip * seq_length:seq_skip]
                mask_blocks.append(mask_block)

                if self.have_actions:
                    act_block = vid_one_hot_acts[i:i + seq_skip * seq_length:
                                                 seq_skip]
                    action_blocks.append(act_block)

        poses = np.stack(pose_blocks, axis=0)
        masks = np.stack(mask_blocks, axis=0)

        if self.have_actions:
            actions = np.stack(action_blocks, axis=0)
        else:
            actions = None

        return poses, masks, actions

    def get_aclass_ds(self, train):
        assert self.have_actions, \
            "can't make action classification data without actions"

        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        seq_skip = self.seq_skip
        full_length = self.aclass_full_length
        act_length = self.aclass_act_length
        gap = self.gap

        rv = []

        for vid_idx in vids.index:
            feats = vids['poses'][vid_idx]
            actions = vids['actions'][vid_idx]

            # need T*D features (T time, D dimensionality of features)
            assert feats.ndim == 2, feats.shape
            # actions should be single array of action numbers
            assert actions.ndim == 1, actions.shape
            assert len(feats) == len(actions)
            offset = (full_length - act_length) * seq_skip
            all_runs = _runs(actions[offset:])
            # TODO: check what fraction of these are too short. Per Anoop's
            # suggestion, it probably makes sense to include a bit of context
            # beforehand. That means that I don't have to guarantee that the
            # entire sequence has the same action---just that the last few
            # frames do.
            for action, start, stop in all_runs:
                length = stop - start
                if length < act_length * seq_skip:
                    continue
                stop += offset
                range_end = stop - (seq_skip * full_length) + 1
                for sub_start in range(start, range_end, gap):
                    # no need to temporally downsample; features have already
                    # been temporally downsampled
                    these_feats = feats[sub_start:sub_start+full_length*seq_skip:seq_skip]
                    assert len(these_feats) == full_length
                    suffix = actions[sub_start:sub_start+full_length*seq_skip:seq_skip][-act_length:]
                    assert (suffix == action).all(), \
                        "suffix %s should be all action %d" % (suffix, action)
                    rv.append((these_feats, action))

        return rv

    def get_completion_ds(self, train):
        # choose non-overlapping seqeuences for completion dataset
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        seq_skip = self.seq_skip
        completion_length = self.completion_length
        assert completion_length > 0

        completions = []

        for vid_idx in vids.index:
            norm_poses = vids['poses'][vid_idx]
            mask = vids['mask'][vid_idx]
            vid_name = vids['vid_name'][vid_idx]

            if self.have_actions:
                actions = vids['actions'][vid_idx]
                one_hot_acts = np.zeros((len(norm_poses), self.num_actions),
                                        dtype='float32')
                one_hot_acts[(range(len(actions)), actions)] = 1

            range_bound = 1 + len(norm_poses) - seq_skip * completion_length
            block_skip = seq_skip * completion_length

            for i in range(0, range_bound, block_skip):
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
                    'skip': seq_skip,
                }
                if self.have_actions:
                    action_block = one_hot_acts[i:endpoint:seq_skip]
                    completion_block['actions'] = action_block
                    assert len(action_block) == len(pose_block)
                completions.append(completion_block)

        return completions

    def reconstruct_poses(self, rel_block, vid_name):
        assert rel_block.ndim == 3 \
            and rel_block.shape[-1] == 2 * len(self.parents), \
            rel_block.shape

        # 1) Account for self.std, self.mean
        rel_block = rel_block * self.std[None, ...] \
            + self.mean[None, ...]

        # 2) Undo preprocess_sequence (except for head thing, which was a
        #    one-way trip)
        vid_idx = np.argwhere(self.videos['vid_name'] == vid_name)
        pp_offset = self.videos['pp_offset'][vid_idx]
        pp_scale = self.videos['pp_scale'][vid_idx]
        block = _reconstruct_poses(rel_block, self.parents, pp_offset,
                                   pp_scale)

        assert block.shape == rel_block.shape[:2] + (2, len(self.parents)), \
            block.shape

        return block
