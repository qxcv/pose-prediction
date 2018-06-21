#!/usr/bin/env python3
"""Turn original dataset sequences into videos. Super duper cut-and-pasted."""

import argparse
import json
import os

import h5py

import numpy as np

import matplotlib.pyplot as plt

import addpaths  # noqa
from plot_2d_seqs import draw_poses
from common_pp.completion_video_common import load_sorted_paths, \
    alignment_constant

H5_PATH = './mpii_ca2.h5'
FRAME_DIR = '/data/home/cherian/MPII/Cheng-MPII-Pose-Action/frames/'
FRAME_DIR = '/home/sam/sshfs/paloalto' + FRAME_DIR  # XXX
POSE_DIR = '/home/sam/sshfs/paloalto/etc/cpm-keras/mpii-ca2-mat-poses'  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('vid_name', help='name of video sequence to show')

if __name__ == '__main__':
    args = parser.parse_args()
    vid_name = args.vid_name

    with h5py.File(H5_PATH) as fp:
        poses = fp['/seqs/' + vid_name + '/poses'].value
        action_ids = fp['/seqs/' + vid_name + '/actions'].value
        action_name_map = np.asarray(
            json.loads(
                fp['/action_names'].value.tostring().decode('utf8')))
        parents = fp['/parents'].value

    pose_seqs = poses[None, ...]
    action_labels = action_name_map[action_ids]
    frame_paths = load_sorted_paths(os.path.join(FRAME_DIR, vid_name))
    seq_names = ['True poses']
    all_mat_pose_paths = load_sorted_paths(os.path.join(POSE_DIR, vid_name))
    mat_fst_pose_path = all_mat_pose_paths[0]
    with h5py.File(mat_fst_pose_path) as fp:
        # gives us 2*14
        ref_pose = fp['pose'].value[:, :8].astype('float')
    alpha, beta = alignment_constant(pose_seqs[0, 0], ref_pose)

    pose_seqs = pose_seqs * alpha + beta[None, None, :, None]

    assert len(action_labels) == len(poses)
    print('Action names: ' + ', '.join(action_name_map))
    print('Actions: ' + ', '.join(action_labels))

    # important not to let return value be gc'd (anims won't run otherwise!)
    anims = draw_poses(
        'Poses in %s' % vid_name,
        parents,
        pose_seqs,
        frame_paths=[frame_paths] * 3,
        subplot_titles=seq_names,
        fps=50 / 9.0,
        action_labels=action_labels)
    plt.show()
