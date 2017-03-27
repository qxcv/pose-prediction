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

# CPM joints: head, neck, right shoulder/elbow/wrist (PC), left
# shoulder/elbow/wrist (PC)
TO_CPM = [15, 13, 25, 26, 27, 17, 18, 19]
PARENTS = [0, 0, 1, 2, 3, 1, 5, 6]
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
        for fn in listdir(d2_dir):
            # capture full action (can be merged) and cam
            m = re.findall(r'^([^.]+)\.(\d+).cdf$', fn)
            if not m:
                print('Could not match filename "%s"' % fn, file=sys.stderr)
                continue
            assert len(m) == 1
            act_name_full, cam_s = m[0]

            # path to pose CDF file
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
                'vid_id': vid_ident,
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


def process_tasks(to_process, dest):
    with File(dest, 'w') as fp:
        for task in tqdm(to_process):
            seq_prefix = '/seqs/%s/' % task['vid_id']
            poses, scale = load_poses(task['pose_path'])
            actions = np.full((len(poses), ), task['action_id'], dtype='uint8')
            valid = np.ones((poses.shape[0], poses.shape[2]), dtype='bool')
            fp[seq_prefix + 'poses'] = poses
            fp[seq_prefix + 'actions'] = actions
            fp[seq_prefix + 'scale'] = scale
        fp['/parents'] = np.array(PARENTS, dtype=int)
        fp['/action_names'] = np.array(
            [ord(c) for c in dumps(ACTION_NAMES)], dtype='uint8')
        fp['/num_actions'] = len(ACTION_NAMES)


if __name__ == '__main__':
    args = parser.parse_args()
    # I'm splitting the code into "collect" and "process" loops so that I can
    # use tqdm to monitor the hard stuff.
    to_process = collect_tasks(args.h36m_path)
    process_tasks(to_process, args.dest)
