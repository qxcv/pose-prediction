#!/usr/bin/env python3
"""Conversion script for 2D H3.6M dataset."""

from argparse import ArgumentParser
from json import dumps
from os import path, listdir
import re
import sys

from tqdm import tqdm
from h5py import File
import numpy as np
from spacepy import pycdf

from expmap import xyz_to_expmap, bone_lengths
from h36m.parse_meta import joint_hierarchy as get_hierarchy_3d

# 50FPS originally, so ~16.7FPS after keeping every third frame
FRAME_SKIP = 3
# around 3s
CONDITION_LENGTH = 45
# around 5s
TEST_LENGTH = 75
# jump forward by 1s between testing blocks
TEST_GAP = 15
# validate on only one subject
VAL_SUBJECT = 5
# CPM joints: head, neck, right shoulder/elbow/wrist (PC), left
# shoulder/elbow/wrist (PC)
TO_CPM = [15, 13, 25, 26, 27, 17, 18, 19]
PARENTS = [0, 0, 1, 2, 3, 1, 5, 6]
JOINT_NAMES_3D, PARENTS_3D = get_hierarchy_3d()
ACTION_NAMES = [
    # Skip '_ALL'
    'Directions',
    'Discussion',
    'Eating',
    'Greeting',
    'Phoning',
    'Posing',
    'Purchases',
    'Sitting',
    'SittingDown',
    'Smoking',
    'TakingPhoto',
    'Waiting',
    'Walking',
    'WalkingDog',
    'WalkTogether',
]
CANON_MAP = {
    # get rid of stupid duplicates
    'WalkDog': 'WalkingDog',
    'Photo': 'TakingPhoto',
}
CANON_MAP.update((v, v) for v in ACTION_NAMES)

parser = ArgumentParser()
parser.add_argument('h36m_path', help='path to H3.6M dataset')
parser.add_argument('dest', help='path for HDF5 output file')


def collect_tasks(h36m_path):
    to_process = []
    for fn in listdir(h36m_path):
        matches = re.findall(r'^S(\d+)$', fn)
        if not matches:
            continue
        subnum_s, = matches
        subnum = int(subnum_s)
        d2_dir = path.join(args.h36m_path, fn, 'MyPoseFeatures',
                           'D2_Positions')
        d3_dir = path.join(args.h36m_path, fn, 'MyPoseFeatures',
                           'D3_Positions_mono_universal')
        for fn in listdir(d2_dir):
            # capture full action (can be merged) and cam
            m = re.findall(r'^([^.]+)\.(\d+).cdf$', fn)
            if not m:
                print('Could not match filename "%s"' % fn, file=sys.stderr)
                continue
            assert len(m) == 1
            act_name_full, cam_s = m[0]

            # for 3D skeleton
            skeleton_path = path.join(d3_dir, fn)

            # path to 2D pose CDF file
            pose_path = path.join(d2_dir, fn)

            # simple action name
            init_full_name = act_name_full.split()[0]
            if init_full_name not in CANON_MAP:
                print(
                    'Could not match action "%s"' % act_name_full,
                    file=sys.stderr)
                continue
            merged_action_id = ACTION_NAMES.index(CANON_MAP[init_full_name])

            # path to video of person doing action
            vid_path = path.join(args.h36m_path, fn, 'Videos',
                                 '%s.%s.mp4' % (act_name_full, cam_s))

            # unique identifier for the HDF5 file
            vid_ident = 's:%d:%s:%s' % (subnum, act_name_full, cam_s)
            to_process.append({
                'subject': subnum,
                'action_id': merged_action_id,
                'vid_path': vid_path,
                'pose_path': pose_path,
                'skeleton_path': skeleton_path,
                'vid_id': vid_ident,
                'is_train': subnum != VAL_SUBJECT
            })
    return to_process


def load_poses(cdf_path):
    with pycdf.CDF(cdf_path) as fp:
        pvar = fp['Pose']
        assert len(pvar) == 1, pvar.shape
        flat_arr = pvar[0]
    assert flat_arr.ndim == 2 and flat_arr.shape[1] == 64, flat_arr.shape
    poses = flat_arr.reshape((flat_arr.shape[0], flat_arr.shape[1] // 2, 2)) \
                    .transpose([0, 2, 1])
    # returns T*2*J array
    assert poses.shape[1:] == (2, 32), poses.shape
    cpm_poses = poses[:, :, TO_CPM]
    # use mean hip-shoulder distance as a scale factor
    # L/R H/S is 6/25, R/L H/S is 1/17
    lh_rs_dist = np.linalg.norm(poses[:, :, 6] - poses[:, :, 25], axis=1)
    rh_ls_dist = np.linalg.norm(poses[:, :, 1] - poses[:, :, 17], axis=1)
    dists = np.concatenate([lh_rs_dist, rh_ls_dist])
    assert len(dists) == 2 * len(poses)
    scale = np.mean(dists)
    cpm_poses = cpm_poses.astype('float32')
    return cpm_poses, scale


def load_skeletons(cdf_path):
    with pycdf.CDF(cdf_path) as fp:
        pvar = fp['Pose']
        assert len(pvar) == 1, pvar.shape
        flat_arr = pvar[0]
    assert flat_arr.ndim == 2 and flat_arr.shape[1] == 96, flat_arr.shape
    xyz_skeletons = flat_arr.reshape((len(flat_arr), -1, 3))
    exp_skeletons = xyz_to_expmap(xyz_skeletons, PARENTS_3D)
    return xyz_skeletons, exp_skeletons


def h5_json_encode(data):
    """Turns rich Python data structure into array of bytes so that it can be
    stuffed in an HDF5 file."""
    char_codes = [ord(c) for c in dumps(data)]
    return np.array(char_codes, dtype='uint8')


def process_tasks(to_process, dest):
    all_bone_lengths = []

    with File(dest, 'w') as fp:
        for task in tqdm(to_process):
            # 2D first
            seq_prefix = '/seqs/%s/' % task['vid_id']
            poses, scale = load_poses(task['pose_path'])
            actions = np.full((len(poses), ), task['action_id'], dtype='uint8')
            # valid = np.ones((poses.shape[0], poses.shape[2]), dtype='bool')
            is_train = task['is_train']
            fp[seq_prefix + 'poses'] = poses
            fp[seq_prefix + 'actions'] = actions
            fp[seq_prefix + 'scale'] = np.full((len(poses),), scale)
            fp[seq_prefix + 'is_train'] = is_train

            # now 3D
            seq_3d_prefix = '/seqs3d/%s/' % task['vid_id']
            xyz_skeletons, exp_skeletons = load_skeletons(
                task['skeleton_path'])
            all_bone_lengths.append(bone_lengths(xyz_skeletons, PARENTS_3D))
            fp[seq_3d_prefix + 'skeletons'] = exp_skeletons.astype('float32')
            fp[seq_3d_prefix + 'is_train'] = is_train

        fp['/parents'] = np.array(PARENTS, dtype=int)
        fp['/action_names'] = h5_json_encode(ACTION_NAMES)
        fp['/num_actions'] = len(ACTION_NAMES)

        fp['/frame_skip'] = FRAME_SKIP
        fp['/eval_condition_length'] = CONDITION_LENGTH
        fp['/eval_test_length'] = TEST_LENGTH
        fp['/eval_seq_gap'] = TEST_GAP

        fp['/joint_names_3d'] = h5_json_encode(JOINT_NAMES_3D)
        fp['/parents_3d'] = np.array(PARENTS_3D, dtype='uint8')
        all_bone_lengths = np.concatenate(all_bone_lengths, axis=0)
        const_bone_lengths = np.median(all_bone_lengths, axis=0)
        fp['/bone_lengths_3d'] = const_bone_lengths.astype('float32')


if __name__ == '__main__':
    args = parser.parse_args()
    # I'm splitting the code into "collect" and "process" loops so that I can
    # use tqdm to monitor the hard stuff.
    to_process = collect_tasks(args.h36m_path)
    process_tasks(to_process, args.dest)
