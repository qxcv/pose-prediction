#!/usr/bin/env python3

"""Conversion script for MPII Cooking Activities 2 dataset."""

# TODO: Figure out action names! They'll be in the attributes file somewhere.

from argparse import ArgumentParser
from glob import glob
from h5py import File
from json import dumps
import numpy as np
from os import path
import sys
from textwrap import wrap
from tqdm import tqdm
from multiprocessing import Pool

# I'm pretty sure this is the same as IkeaDB, since the poses were produced by
# the same model (CPM).
PARENTS = [0, 0, 1, 2, 3, 1, 5, 6]
# Ignore things that aren't upper-body joints
GOOD_JOINTS = range(8)
assert len(GOOD_JOINTS) == len(PARENTS)
# ACTION_NAMES = []

parser = ArgumentParser()
# Anoop provided the pose directory. Check
# /data/home/cherian/MPII/Cheng-MPII-Pose-Action/detected_poses/
parser.add_argument('pose_path', help='path to MPII pose dir (from CPM)')
# This is shipped with Cooking Activities 2. See
# /data/home/sam/mpii-cooking-2/attributesAnnotations_MPII-Cooking-2.mat (I
# think that's the right one, anyway).
parser.add_argument('attr_path', help='path to MPII attributes file (.mat)')
parser.add_argument('dest', help='path for HDF5 output file')


def load_seq(mat_dir):
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

    return joints


if __name__ == '__main__':
    args = parser.parse_args()
    dir_list = glob(path.join(args.pose_path, 's*-d*-cam-*'))
    with File(args.dest, 'w') as fp:
        skipped = []
        with Pool() as p:
            seq_iter = p.imap(load_seq, dir_list)
            zipper = zip(dir_list, seq_iter)
            for dir_path, joints in tqdm(zipper, total = len(dir_list)):
                joints = load_seq(dir_path)
                id_str = path.basename(dir_path)
                if joints is None:
                    skipped.append(id_str)
                    continue
                # action_id = ACTION_NAMES.index(action)
                prefix = '/seqs/' + id_str + '/'
                fp[prefix + 'poses'] = joints
                # fp[prefix + 'actions'] = np.full((len(joints),), action_id)
        fp['/parents'] = np.array(PARENTS, dtype=int)
        # fp['/action_names'] = np.array([ord(c) for c in dumps(ACTION_NAMES)],
        #                                dtype='uint8')
        # fp['/num_actions'] = len(ACTION_NAMES)
        if skipped:
            print('WARNING: skipped %i seq(s) due to scale:' % len(skipped),
                  file=sys.stderr)
            print('\n'.join(wrap(', '.join(skipped))))
