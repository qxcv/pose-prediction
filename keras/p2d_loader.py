"""Utilities for loading 2D pose sequences"""

import numpy as np
from scipy import stats, signal
import json
import h5py
import pandas as pd


def gauss_filter(seq, sigma, filter_width=None):
    """Filter a 2D signal, where first dimension is time, and second is data
    channels."""
    if len(seq) <= 1:
        # can't do any meaningful filtering
        return seq
    if filter_width is None:
        filter_width = min(int(np.ceil(4 * sigma)), len(seq))
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


def preprocess_sequence(poses,
                        skip,
                        parents,
                        smooth_sigma=False,
                        head_vel=True,
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
        if head_vel:
            # Record delta from frame which is <skip> steps in the past
            relpose[skip:, :, 0] \
                = norm_poses[skip:, :, 0] - norm_poses[:-skip, :, 0]
        else:
            relpose[:, :, 0] = norm_poses[:, :, 0]
        # Other norm_poses record delta from parents
        for jt in range(1, len(parents)):
            pa = parents[jt]
            relpose[:, :, jt] = norm_poses[:, :, jt] - norm_poses[:, :, pa]

        # Collapse in last two dimensions, interleaving X and Y coordinates
        shaped = relpose.reshape((relpose.shape[0], -1))
    else:
        shaped = poses.reshape((norm_poses.shape[0], -1))

    return shaped, offset, scale


def _reconstruct_poses(flat_poses,
                       parents,
                       pp_offset=None,
                       pp_scale=None,
                       head_vel=True):
    """Undo parent-relative joint transform. Will not undo the uniform scaling
    applied to each sequence."""
    # shape of poses shold be (num training samples)*(time)*(flattened
    # dimensions)
    assert flat_poses.ndim == 3, flat_poses.shape
    if pp_offset is not None or pp_scale is not None:
        assert pp_offset is not None
        assert pp_scale is not None
        assert pp_offset.size == 2, 'expect offset to be XY vec'
        assert np.asarray(pp_scale).size == 1, 'expect scale to be scalar'

    # rel_poses is a 4D array: (num samples)*T*(XY)*J
    rel_poses = flat_poses.reshape(flat_poses.shape[:2] + (2, -1))
    true_poses = np.zeros_like(rel_poses)
    N, T, Dxy, J = true_poses.shape
    assert Dxy == 2
    assert len(parents) == J

    # start by restoring head from velocity param, if necessary
    if head_vel:
        rel_heads = rel_poses[:, :, :, 0]
        true_heads = np.cumsum(rel_heads, axis=1)
        true_poses[:, :, :, 0] = true_heads
    else:
        true_poses[:, :, :, 0] = rel_poses[:, :, :, 0]

    # now reconstruct remaining joints from parents
    for joint in range(1, len(parents)):
        parent = parents[joint]
        parent_pos = true_poses[:, :, :, parent]
        offsets = rel_poses[:, :, :, joint]
        true_poses[:, :, :, joint] = parent_pos + offsets

    if pp_offset is not None:
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
    is_3d = False

    def __init__(self,
                 data_file_path,
                 seq_length,
                 gap=1,
                 have_actions=True,
                 completion_length=None,
                 relative=True,
                 aclass_full_length=128,
                 aclass_act_length=8,
                 remove_head=False,
                 head_vel=True):
        """Preprocess an open HDF5 file which has been formatted to hold 2D
        poses.

        :param data_file_path: Path to HDF5 file.
        :param seq_length: Length of output sequences.
        :param gap: How far to skip forward between each sampled sequence.
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
        self.gap = gap
        self.have_actions = have_actions
        self.relative = relative
        self.remove_head = remove_head
        self.completion_length = completion_length
        self.aclass_full_length = aclass_full_length
        self.aclass_act_length = 8
        self.head_vel = head_vel
        videos_list = []

        with h5py.File(data_file_path, 'r') as fp:
            self.parents = fp['/parents'].value
            self.frame_skip = int(fp['/frame_skip'].value)
            self.eval_condition_length = int(fp['eval_condition_length'].value)
            self.eval_test_length = int(fp['eval_test_length'].value)
            self.eval_seq_gap = int(fp['eval_seq_gap'].value)
            hsa_key = 'has_sparse_annos'
            self.has_sparse_annos = hsa_key in fp and bool(fp[hsa_key].value)
            if 'pck_joints' in fp:
                pck_limb_str = fp['/pck_joints']\
                                 .value \
                                 .tostring() \
                                 .decode('utf8')
                self.pck_joints = json.loads(pck_limb_str)
            else:
                self.pck_joints = {'all': list(range(len(self.parents)))}

            if self.remove_head:
                # head removal matrix
                hrm_mat, self.parents = head_remover(self.parents)
                if self.pck_joints is not None:
                    for k, v in list(self.pck_joints.items()):
                        new_v = [x - 1 for x in v if x != 0]
                        if not new_v:
                            del self.pck_joints[k]
                        else:
                            self.pck_joints[k] = new_v
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

            for vid_name in vid_names:
                orig_poses = fp['/seqs/' + vid_name + '/poses'].value

                sfact_path = '/seqs/' + vid_name + '/scale'
                if sfact_path in fp:
                    scales = fp[sfact_path].value
                else:
                    scales = np.ones((len(orig_poses), ))

                assert scales.ndim == 1
                assert len(scales) == len(orig_poses)

                # we divide by manually annotated scales BEFORE we do the
                # preprocess_sequence pass
                orig_poses = orig_poses.astype('float32') / scales.reshape(
                    (-1, 1, 1))
                if np.any(np.isnan(orig_poses)) or np.any(
                        np.abs(orig_poses) > 1e5):
                    print('Rejecting %s (invalid or too big)' % vid_name)
                    continue
                # don't both with relative poses
                norm_poses, pp_offset, pp_scale = preprocess_sequence(
                    orig_poses,
                    self.frame_skip,
                    self.parents if self.relative else None,
                    smooth_sigma=2,
                    head_vel=head_vel,
                    hrm_mat=hrm_mat)
                norm_poses = norm_poses.astype('float32')
                # make sure we don't reuse poses in native parameterisation
                del orig_poses

                if self.have_actions:
                    actions = fp['/seqs/' + vid_name + '/actions'].value
                    assert len(actions) == len(norm_poses)

                # XXX: this is a terrible idea! What if the manipulations we do
                # in preprocess_sequence destory some information relevant to
                # validity?
                if '/seqs/' + vid_name + '/valid' in fp:
                    mask = fp['/seqs/' + vid_name + '/valid'].value
                    mask = mask.reshape((mask.shape[0], -1)).astype('float32')
                    assert mask.shape == norm_poses.shape, \
                        "mask should be %s, but was %s" % (norm_poses.shape,
                                                           mask.shape)
                else:
                    mask = np.ones_like(norm_poses, dtype='float32')

                is_train = fp['/seqs/' + vid_name + '/is_train'].value

                vid_meta = {
                    'pp_offset': pp_offset,
                    'pp_scale': pp_scale,
                    'poses': norm_poses,
                    'vid_name': vid_name,
                    'mask': mask,
                    'is_val': not is_train,
                    # pp_scale is produced by preprocess_sequence without
                    # reference to original sequence scale. scales is given by
                    # the dataset writing code (probably represents
                    # hip-shoulder distance)
                    'scales': scales
                }
                if self.have_actions:
                    vid_meta['actions'] = actions
                if self.has_sparse_annos:
                    is_true_pose = fp['/seqs/' +
                                      vid_name + '/is_true_pose'].value
                    assert is_true_pose.ndim == 1 and len(is_true_pose) == len(
                        norm_poses)
                    vid_meta['is_true_pose'] = is_true_pose.astype(bool)

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
        pms_zip = zip(self.videos['poses'], self.videos['mask'],
                      self.videos['scales'])
        for poses, mask, scales in pms_zip:
            poses[:] = (poses[:] - self.mean) / self.std
            # TODO: should I leave this in? Does it make things better?
            poses[mask == 0] = 0

        self.dim_obs = self.mean.size

    def get_ds_for_train(self, train, seq_length, gap, discard_shorter):
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        frame_skip = self.frame_skip
        pose_blocks = []
        mask_blocks = []

        for vid_idx in vids.index:
            vid_poses = vids['poses'][vid_idx]
            vid_masks = vids['mask'][vid_idx]
            range_count = len(vid_poses) - frame_skip * discard_shorter + 1

            for i in range(0, range_count, gap):
                end = min(len(vid_poses), i + frame_skip * seq_length)
                pose_block = vid_poses[i:end:frame_skip]
                block_length = len(pose_block)
                assert block_length <= seq_length
                pads = [(0, seq_length - block_length), (0, 0)]
                pose_block_padded = np.pad(
                    pose_block, pads, mode='constant').astype('float32')
                assert np.all(pose_block_padded[:block_length] == pose_block)
                pose_blocks.append(pose_block_padded)

                # fill out the sequence with some masked time steps
                mask_block = vid_masks[i:end:frame_skip]
                pads = [(0, seq_length - block_length)] \
                    + [(0, 0)] * (mask_block.ndim - 1)
                mask_block_padded = np.pad(
                    mask_block, pads, mode='constant', constant_values=0)
                mask_blocks.append(mask_block_padded.astype('float32'))

        poses = np.stack(pose_blocks, axis=0)
        del pose_blocks
        masks = np.stack(mask_blocks, axis=0)
        del mask_blocks
        assert poses.ndim == 3 and poses.shape[1] == seq_length, \
            poses.shape
        assert poses.shape == masks.shape, (poses.shape, masks.shape)

        return poses, masks

    def get_pose_ds(self, train):
        # poses should be N*T*D, actions should be N*T*A if they exist
        # (one-hot), mask should be N*T*D
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        frame_skip = self.frame_skip
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
                vid_one_hot_acts = np.zeros(
                    (len(vid_actions), self.num_actions), dtype='float32')
                inds = (np.arange(len(vid_one_hot_acts)), vid_actions)
                vid_one_hot_acts[inds] = 1

            range_count = len(vid_poses) - frame_skip * seq_length + 1

            for i in range(0, range_count, gap):
                end = i + frame_skip * seq_length
                pose_block = vid_poses[i:end:frame_skip]
                pose_blocks.append(pose_block)
                mask_block = vid_mask[i:end:frame_skip]
                mask_blocks.append(mask_block)

                if self.have_actions:
                    act_block = vid_one_hot_acts[i:i + frame_skip * seq_length:
                                                 frame_skip]
                    action_blocks.append(act_block)

        poses = np.stack(pose_blocks, axis=0)
        masks = np.stack(mask_blocks, axis=0)

        if self.have_actions:
            actions = np.stack(action_blocks, axis=0)
        else:
            actions = None

        return poses, masks, actions

    def get_ds_for_eval(self, train, discard_no_annos=True):
        # use eval config described on experiment protocol page
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        cond_len = self.eval_condition_length
        test_len = self.eval_test_length
        tot_len = cond_len + test_len
        eval_seq_gap = self.eval_seq_gap
        frame_skip = self.frame_skip
        out_seqs = []
        scale_seqs = []
        seq_ids = []
        frame_num_blocks = []
        if self.has_sparse_annos:
            valid_seqs = []

        for vid_idx in vids.index:
            vid_poses = vids['poses'][vid_idx]
            vid_scales = vids['scales'][vid_idx]
            if self.has_sparse_annos:
                true_pose_mask = vids['is_true_pose'][vid_idx]
            range_count = len(vid_poses) - frame_skip * tot_len + 1
            seq_id = vids['vid_name'][vid_idx]

            # TODO: need to handle mask, etc., if I'm going to run on Penn
            # Action

            for i in range(0, range_count, eval_seq_gap * frame_skip):
                end = i + frame_skip * tot_len
                assert end <= len(vid_poses)
                if self.has_sparse_annos:
                    valid_block = true_pose_mask[i:end:frame_skip]
                    if discard_no_annos and not valid_block[cond_len:].any():
                        continue
                    valid_seqs.append(valid_block)
                pose_block = vid_poses[i:end:frame_skip]
                out_seqs.append(pose_block)
                scale_block = vid_scales[i:end:frame_skip]
                scale_seqs.append(scale_block)
                seq_ids.append(seq_id)
                frame_num_blocks.append(list(range(i, end, frame_skip)))

        all_seqs = np.stack(out_seqs, axis=0)
        all_nums = np.stack(frame_num_blocks, axis=0)
        assert all_seqs.shape[1] == tot_len
        conditioning = all_seqs[:, :cond_len]
        prediction = all_seqs[:, cond_len:]
        all_scales = np.stack(scale_seqs, axis=0)
        prediction_scales = all_scales[:, cond_len:]
        prediction_nums = all_nums[:, cond_len:]
        conditioning_scales = all_scales[:, :cond_len]
        conditioning_nums = all_nums[:, :cond_len]
        assert prediction_nums.shape == prediction.shape[:2]
        rv = {
            'seq_ids': np.asarray(seq_ids),
            'conditioning': conditioning,
            'conditioning_scales': conditioning_scales,
            'conditioning_frame_nums': conditioning_nums,
            'prediction': prediction,
            'prediction_scales': prediction_scales,
            'prediction_frame_nums': prediction_nums
        }
        if self.has_sparse_annos:
            all_valids = np.stack(valid_seqs, axis=0)
            prediction_valids = all_valids[:, cond_len:]
            rv['prediction_valids'] = prediction_valids

        return rv

    def get_aclass_ds(self, train):
        assert self.have_actions, \
            "can't make action classification data without actions"

        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        frame_skip = self.frame_skip
        full_length = self.aclass_full_length
        act_length = self.aclass_act_length
        gap = self.gap

        rv = []

        for vid_idx in vids.index:
            vid_name = vids['vid_name'][vid_idx]
            feats = vids['poses'][vid_idx]
            actions = vids['actions'][vid_idx]
            mask = vids['mask'][vid_idx]
            if len(actions) < full_length:
                print('Skipping %s becuase it is too short (only %d frames)!' %
                      (vids['vid_name'][vid_idx], len(actions)))
                continue

            # need T*D features (T time, D dimensionality of features)
            assert feats.ndim == 2, feats.shape
            # actions should be single array of action numbers
            assert actions.ndim == 1, actions.shape
            assert len(feats) == len(actions)
            offset = (full_length - act_length) * frame_skip
            all_runs = _runs(actions[offset:])

            # TODO: check what fraction of these are too short. Per Anoop's
            # suggestion, it probably makes sense to include a bit of context
            # beforehand. That means that I don't have to guarantee that the
            # entire sequence has the same action---just that the last few
            # frames do.
            for action, start, stop in all_runs:
                length = stop - start
                if length < act_length * frame_skip:
                    continue
                stop += offset
                range_end = stop - (frame_skip * full_length) + 1

                for sub_start in range(start, range_end, gap):
                    # no need to temporally downsample; features have already
                    # been temporally downsampled
                    these_feats = feats[sub_start:sub_start + full_length *
                                        frame_skip:frame_skip]
                    assert len(these_feats) == full_length

                    these_actions = actions[sub_start:sub_start + full_length *
                                            frame_skip:frame_skip]
                    action_suffix = these_actions[-act_length:]
                    assert len(these_actions) == full_length
                    assert len(action_suffix) == act_length
                    assert (action_suffix == action).all(), \
                        "suffix %s should be all action %d" \
                        % (action_suffix, action)

                    this_mask = mask[sub_start:sub_start + full_length *
                                     frame_skip:frame_skip]
                    assert len(this_mask) == full_length

                    rv.append({
                        'poses': these_feats,
                        'mask': this_mask,
                        # 'actions' is a vector of actinos for the entire
                        # sequence. 'action_label' is the action at the end
                        # (which the action classifier must predict!)
                        'actions': these_actions,
                        'action_label': action,
                        'vid_name': vid_name
                    })

        return rv

    def get_completion_ds(self, train):
        # choose non-overlapping seqeuences for completion dataset
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        frame_skip = self.frame_skip
        completion_length = self.completion_length
        assert completion_length > 0

        completions = []

        for vid_idx in vids.index:
            norm_poses = vids['poses'][vid_idx]
            mask = vids['mask'][vid_idx]
            vid_name = vids['vid_name'][vid_idx]

            if self.have_actions:
                actions = vids['actions'][vid_idx]
                one_hot_acts = np.zeros(
                    (len(norm_poses), self.num_actions), dtype='float32')
                one_hot_acts[(range(len(actions)), actions)] = 1

            range_bound = 1 + len(norm_poses) - frame_skip * completion_length
            block_skip = frame_skip * completion_length

            for i in range(0, range_bound, block_skip):
                endpoint = i + frame_skip * completion_length
                pose_block = norm_poses[i:endpoint:frame_skip]
                mask_block = mask[i:endpoint:frame_skip]
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
                    'skip': frame_skip,
                }
                if self.have_actions:
                    action_block = one_hot_acts[i:endpoint:frame_skip]
                    completion_block['actions'] = action_block
                    assert len(action_block) == len(pose_block)
                completions.append(completion_block)

        return completions

    def reconstruct_poses(self, rel_block, vid_names=None, frame_inds=None):
        assert rel_block.ndim == 3 \
            and rel_block.shape[-1] == 2 * len(self.parents), \
            rel_block.shape

        # 1) Account for self.std, self.mean
        rel_block = rel_block * self.std[None, ...] \
            + self.mean[None, ...]

        # 2) Undo preprocess_sequence (except for head removal thing;
        #    decapitation is a one-way trip)
        if vid_names is not None:
            vid_names = np.asarray(vid_names)
            ex_shape = (len(rel_block),)
            assert vid_names.shape == ex_shape, \
                "Expected vid_names to be shape %s, but was %s" \
                % (ex_shape, vid_names.shape)
            block_rows = []
            for row in range(rel_block.shape[0]):
                row_name = vid_names[row]
                vid_idx = int(np.argwhere(self.videos['vid_name'] == row_name))
                # this undoes the automatic scaling
                pp_offset = self.videos['pp_offset'][vid_idx]
                pp_scale = self.videos['pp_scale'][vid_idx]
                block_row = _reconstruct_poses(
                    rel_block[row:row+1],
                    self.parents,
                    pp_offset,
                    pp_scale,
                    head_vel=self.head_vel)

                # now undo the manual scaling
                scales = self.videos['scales'][vid_idx]
                if frame_inds is None:
                    vid_scale = scales.mean()
                    if not np.all(np.abs(scales == vid_scale) < 1e-5):
                        print(('WARNING! Using one scale for entire video %s, '
                               'but scales differ between frames!') % row_name)
                    block_row = block_row * vid_scale
                else:
                    # otherwise we can use frame indices to figure out what
                    # right scales are
                    these_inds = frame_inds[row]
                    right_scales = scales[these_inds]
                    block_row = block_row / right_scales[None, :, None, None]

                block_rows.append(block_row)
            block = np.concatenate(block_rows, axis=0)
        else:
            block = _reconstruct_poses(
                rel_block, self.parents, head_vel=self.head_vel)

        assert block.shape == rel_block.shape[:2] + (2, len(self.parents)), \
            block.shape

        return block


class P3DDataset(object):
    """Mimics P2DDataset, but is specialised to 3D data. Much less complicated
    because it doesn't need to deal with preprocessing, joint validity
    (everything is assumed okay) or actions, for now."""
    is_3d = True

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        videos_list = []

        with h5py.File(data_file_path, 'r') as fp:
            vid_names = list(fp['seqs3d'])
            self.parents = fp['parents_3d'].value
            self.eval_condition_length = fp['eval_condition_length'].value
            self.eval_test_length = fp['eval_test_length'].value
            self.frame_skip = fp['frame_skip'].value
            self.eval_seq_gap = fp['eval_seq_gap'].value

            for vid_name in vid_names:
                skeletons = fp['/seqs3d/' + vid_name + '/skeletons'].value
                skeletons_flat = skeletons \
                    .astype('float32') \
                    .reshape((skeletons.shape[0], -1))
                is_train = fp['/seqs3d/' + vid_name + '/is_train'].value

                vid_meta = {
                    'skeletons': skeletons_flat,
                    'vid_name': vid_name,
                    'is_val': not is_train
                }

                videos_list.append(vid_meta)

        self.videos = pd.DataFrame.from_records(videos_list)

        train_vids = self.videos[~self.videos.is_val]
        flat_skeletons = np.concatenate(
            train_vids.skeletons.as_matrix(), axis=0)
        assert flat_skeletons.ndim == 2, flat_skeletons.shape
        self.mean = flat_skeletons.mean(axis=0).reshape((1, -1))
        self.std = flat_skeletons.std(axis=0).reshape((1, -1))
        std_mask = self.std < 1e-5
        self.std[std_mask] = 1
        for skeletons in self.videos['skeletons']:
            skeletons[:] = (skeletons[:] - self.mean) / self.std

        self.dim_obs = self.mean.size

    def reconstruct_skeletons(self, data):
        """Turn flattened data items back into properly-shaped skeletons,
        undoing any mean manipulation along the way."""
        unnorm = data * self.std + self.mean
        J = len(self.parents)
        right_shape = unnorm.reshape(unnorm.shape[:-1] + (J, 3))
        return right_shape

    def get_ds_for_train(self, train, seq_length, gap, discard_shorter):
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        frame_skip = self.frame_skip
        skeleton_blocks = []
        mask_blocks = []

        for vid_idx in vids.index:
            vid_skeletons = vids['skeletons'][vid_idx]
            range_count = len(vid_skeletons) - frame_skip * discard_shorter + 1

            for i in range(0, range_count, gap):
                end = min(len(vid_skeletons), i + frame_skip * seq_length)
                skeleton_block = vid_skeletons[i:end:frame_skip]
                block_length = len(skeleton_block)
                assert block_length <= seq_length
                pads = [(0, seq_length - block_length), (0, 0)]
                skeleton_block_padded = np.pad(
                    skeleton_block, pads, mode='constant').astype('float32')
                assert np.all(
                    skeleton_block_padded[:block_length] == skeleton_block)
                skeleton_blocks.append(skeleton_block_padded)

                # fill out the sequence with some masked time steps
                mask_block = np.zeros_like(
                    skeleton_block_padded, dtype='float32')
                mask_block[:block_length] = 1
                mask_blocks.append(mask_block)

        skeletons = np.stack(skeleton_blocks, axis=0)
        del skeleton_blocks
        masks = np.stack(mask_blocks, axis=0)
        del mask_blocks
        assert skeletons.ndim == 3 and skeletons.shape[1] == seq_length, \
            skeletons.shape
        assert skeletons.shape == masks.shape, (skeletons.shape, masks.shape)

        return skeletons, masks

    def get_ds_for_eval(self, train):
        # use eval config described on experiment protocol page
        if train:
            vids = self.videos[~self.videos.is_val]
        else:
            vids = self.videos[self.videos.is_val]

        cond_len = self.eval_condition_length
        test_len = self.eval_test_length
        tot_len = cond_len + test_len
        eval_seq_gap = self.eval_seq_gap
        frame_skip = self.frame_skip
        out_seqs = []

        for vid_idx in vids.index:
            vid_skeletons = vids['skeletons'][vid_idx]
            range_count = len(vid_skeletons) - frame_skip * tot_len + 1

            for i in range(0, range_count, eval_seq_gap * frame_skip):
                end = i + frame_skip * tot_len
                assert end <= len(vid_skeletons)
                skeleton_block = vid_skeletons[i:end:frame_skip]
                assert len(skeleton_block) == tot_len
                out_seqs.append(skeleton_block)

        all_seqs = np.stack(out_seqs, axis=0)
        conditioning = all_seqs[:, :cond_len]
        prediction = all_seqs[:, cond_len:]

        return conditioning, prediction
