#!/usr/bin/env python3

"""Conversion script for MPII Cooking Activities 2 dataset."""

from argparse import ArgumentParser
from glob import glob
from json import dumps
from multiprocessing import Pool
from os import path
import sys
from textwrap import wrap

from tqdm import tqdm

import numpy as np

from h5py import File

# I'm pretty sure this is the same as IkeaDB, since the poses were produced by
# the same model (CPM).
PARENTS = [0, 0, 1, 2, 3, 1, 5, 6]
# Ignore things that aren't upper-body joints
GOOD_JOINTS = range(8)
assert len(GOOD_JOINTS) == len(PARENTS)
# Comments are copied from email to Anoop. Trust them over the code. Make sure
# that you check that all actions (including all actions appearing in merges)
# occur in the dataset. If they don't, then I've probably made a typo.
ACTION_MERGES = {knife_act: 'knife actionV' for knife_act in [
    # cut apartV, cut diceV, cut off endsV, cut out insideV,
    # cut stripesV, cutV, sliceV, chopV
    'cut apartV', 'cut diceV', 'cut off endsV', 'cut out insideV',
    'cut stripesV', 'cutV', 'sliceV', 'chopV'
]}
# XXX: this is broken! Some actions aren't registering, and I can't figure out
# why. Example: s13-d12-cam-002 is claimed to have no actions except n/a, and
# yet it clearly includes some take_out actions. Why is this happening?
ACTION_LIST = [
    'n/a',                # unknown (not in email to Anoop)
    'screw openV',        # 1.  Screw open
    'pourV',              # 2.  Pour
    'screw closeV',       # 3.  Screw close
    'washV',              # 4.  Wash
    'shakeV',             # 5.  Shake
    'knife actionV',      # 6.  Combination of all knife actions
    'addV',               # 7.  Add
    'spiceV',             # 8.  Spice
    'throw in garbageV',  # 9.  Throw in garbage
    'put lidV',           # 10. Put lid
    'take lidV',          # 11. Take lid
    'rip openV',          # 12. Rip open
    'fillV',              # 13. Fill
    'stirV',              # 14. Stir
    'spreadV',            # 15. Spread
    'whipV',              # 16. Whip
    'open eggV',          # 17. Open egg
    'stampV',             # 18. Stamp
]


def load_str_cells(fp, path):
    """Load a Matlab cell array of strings from a path within a h5py File
    object."""
    rv = []
    for r in fp[path].value.flatten():
        # Get around opaque reference objects stored in HDF5 array. I think
        # this is a way of doing string interning or storing heterogenous
        # datatypes (or both?).
        deref = fp[r].value
        as_str = deref.astype('uint8').tobytes().decode('utf8')
        rv.append(as_str)
    return rv


def merge_acts(acts_arr):
    acts_arr = acts_arr.copy()

    # first check that all actions-to-merge occur
    acts_set = set(acts_arr)
    k_set = set(ACTION_MERGES.keys())
    assert k_set.issubset(acts_set), k_set - acts_set

    al_set = set(ACTION_LIST[1:])
    for idx, act in enumerate(acts_arr):
        merged = False
        if act in ACTION_MERGES:
            # XXX
            print('Merging %s into %s' % (act, ACTION_MERGES[act]))
            acts_arr[idx] = act = ACTION_MERGES[act]
            merged = True
        if act not in al_set:
            assert not merged, "about to remove merged action (?! why " \
                "bother with merge if you drop it?)"
            # XXX
            print('Replacing %s with n/a' % act)
            acts_arr[idx] = act = 'n/a'

    # now check that everything in action list (except n/a action) appears in
    # new list
    acts_set = set(acts_arr)
    assert acts_set.issuperset(al_set)

    return acts_arr


def acts_to_one_hot(start_frames, end_frames, acts_by_time, all_acts,
                    num_frames):
    """Turn action names, start times and end times into a series of one-hot
    action vectors. Resolves conflicts (two actions happening at the same time)
    in favour of later action (since I'm not absolutely 100% sure that start
    and end times define closed intervals)."""
    sort_inds = np.argsort(start_frames)
    start_frames = start_frames[sort_inds]
    end_frames = end_frames[sort_inds]
    acts_by_time = acts_by_time[sort_inds]
    int_classes = np.zeros((num_frames,), dtype=int)
    for start, end, act in zip(start_frames, end_frames, acts_by_time):
        act_id, = np.argwhere(all_acts == act)
        int_classes[start:end+1] = act_id

    # now one-hot
    rv = np.zeros((num_frames, len(all_acts)))
    rv[np.arange(num_frames), int_classes] = 1

    return rv


def load_attrs(attr_path):
    with File(attr_path) as fp:
        # the inner fp[] gives us a reference
        # File names are stuff like '/BS/.../tsv/s37-d74-cam-002.tsv'
        vid_names = [path.basename(n)[:-4] for n in
                     load_str_cells(fp, '/annos/annoFileMap')]
        vid_name_to_id = {}
        for idx, vid_name in enumerate(vid_names):
            vid_name_to_id[vid_name] = idx + 1

        # some discrete things are floats rather than ints ;_;
        def fai(p):
            return fp[p].value.flatten().astype(int)
        # Subtract 1 to make them zero-based. They still use closed intervals,
        # IIRC.
        start_frames = fai('/annos/startFrame') - 1
        end_frames = fai('/annos/endFrame') - 1
        vid_ids = fai('/annos/fileId')

        activities = np.asarray(load_str_cells(fp, '/annos/activity'))
        activities = merge_acts(activities)

    return {
        'name_to_id': vid_name_to_id,
        'activities': activities,
        'vid_ids': vid_ids,
        'start_frames': start_frames,
        'end_frames': end_frames
    }


def load_seq(args):
    mat_dir, attr_dict = args
    # id_str will be left-zero-padded
    mat_paths = glob(path.join(mat_dir, '*.mat'))
    to_collate = {}
    for mat_path in mat_paths:
        t = int(path.basename(mat_path).split('.')[0])
        # these are Matlab v7.3 files, so we need to treat them as plain HDF5
        with File(mat_path, 'r') as fp:
            this_pose = fp['/pose'].value.T
        # will be J*2 matrix
        assert this_pose.ndim == 2 and this_pose.shape[1] == 2, this_pose.shape
        # discard lower body junk
        to_collate[t] = this_pose[GOOD_JOINTS]

    joints = np.zeros((len(to_collate), len(GOOD_JOINTS), 2), dtype='float')
    # Times seem to jump forward/backward by fixed amounts (10 frames?) not
    # sure why. Need something like this (probably inelegant implementation
    # here) to put things back in the right order.
    tj = 0
    for t in sorted(to_collate.keys()):
        joints[tj] = to_collate[t]
        tj += 1
    del to_collate

    # Normalise by median upper arm length; no idea whether this works
    # left shoulder/right hip
    hsd_lr = np.linalg.norm(joints[:, 2] - joints[:, 3], axis=1)
    # right shoulder/left hip
    hsd_rl = np.linalg.norm(joints[:, 5] - joints[:, 6], axis=1)
    scale = np.median(np.concatenate((hsd_lr, hsd_rl)))
    # Make sure that scale is sane
    if abs(scale) < 40 or abs(scale) > 400:
        return None
    joints /= scale

    # Need to be T*XY*J
    joints = joints.transpose((0, 2, 1))
    assert joints.shape[1] == 2, joints.shape
    assert joints.shape[0] == len(mat_paths), joints.shape

    # now we make actions
    vid_name = path.basename(mat_dir.rstrip(path.sep))
    vid_mask = attr_dict['vid_ids'] == attr_dict['name_to_id'][vid_name]
    assert vid_mask.ndim == 1 and vid_mask.sum() >= 1, vid_mask.shape
    start_frames = attr_dict['start_frames'][vid_mask]
    end_frames = attr_dict['end_frames'][vid_mask]
    act_names = attr_dict['activities'][vid_mask]
    actions = acts_to_one_hot(start_frames, end_frames, act_names,
                              np.asarray(ACTION_LIST), len(joints))

    return joints, actions


parser = ArgumentParser()
# Anoop provided the pose directory. Check
# /data/home/cherian/MPII/Cheng-MPII-Pose-Action/detected_poses/
parser.add_argument('pose_path', help='path to MPII pose dir (from CPM)')
# This is shipped with Cooking Activities 2. See
# /data/home/sam/mpii-cooking-2/attributesAnnotations_MPII-Cooking-2.mat (I
# think that's the right one, anyway).
parser.add_argument('attr_path', help='path to MPII attributes file (.mat)')
parser.add_argument('dest', help='path for HDF5 output file')


if __name__ == '__main__':
    args = parser.parse_args()
    dir_list = glob(path.join(args.pose_path, 's*-d*-cam-*'))
    attr_dict = load_attrs(args.attr_path)

    with File(args.dest, 'w') as fp:
        skipped = []
        raise RuntimeError('This is all totally broken. Fix it before you do anything else.')
        with Pool() as p:
            # XXX: reinstate pool
            # seq_iter = p.imap(load_seq, ((d, attr_dict) for d in dir_list))
            seq_iter = map(load_seq, ((d, attr_dict) for d in dir_list))
            zipper = zip(dir_list, seq_iter)
            for dir_path, pair in tqdm(zipper, total=len(dir_list)):
                joints, actions = pair
                id_str = path.basename(dir_path)
                if joints is None:
                    skipped.append(id_str)
                    continue
                prefix = '/seqs/' + id_str
                fp[prefix + '/poses'] = joints
                fp[prefix + '/actions'] = actions
        fp['/parents'] = np.array(PARENTS, dtype=int)
        fp['/action_names'] = np.array([ord(c) for c in dumps(ACTION_LIST)],
                                       dtype='uint8')
        fp['/num_actions'] = len(ACTION_LIST)
        if skipped:
            print('WARNING: skipped %i seq(s) due to scale:' % len(skipped),
                  file=sys.stderr)
            print('\n'.join(wrap(', '.join(skipped))))
