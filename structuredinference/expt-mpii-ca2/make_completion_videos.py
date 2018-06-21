#!/usr/bin/env python3
"""Turn completion files into actual SxS videos of the completion, the ground
truth, etc."""

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

FRAME_DIR = '/data/home/cherian/MPII/Cheng-MPII-Pose-Action/frames/'
# FRAME_DIR = '/home/sam/sshfs/paloalto' + FRAME_DIR  # XXX
# POSE_DIR = '/home/sam/sshfs/paloalto/etc/cpm-keras/mpii-ca2-mat-poses'  # XXX
POSE_DIR = '/home/sam/etc/cpm-keras/mpii-ca2-mat-poses'

parser = argparse.ArgumentParser()
parser.add_argument('completion_path', help='path to .json completion file')
parser.add_argument('--vid-dir', type=str, default=None,
    help='save videos to this directory instead of showing poses')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.completion_path) as fp:
        d = json.load(fp)
    vid_name = d['vid_name']
    all_frame_fns = load_sorted_paths(os.path.join(FRAME_DIR, vid_name))
    frame_paths = [all_frame_fns[i] for i in d['frame_inds']]

    pose_seqs = np.stack(
        (d['true_poses'], d['prior_poses'], d['posterior_poses']), axis=0)
    seq_names = ['True poses', 'Prior prediction', 'Posterior prediction']

    all_mat_pose_paths = load_sorted_paths(os.path.join(POSE_DIR, vid_name))
    mat_fst_pose_path = all_mat_pose_paths[d['frame_inds'][0]]
    with h5py.File(mat_fst_pose_path) as fp:
        # gives us 2*14
        ref_pose = fp['pose'].value[:, :8].astype('float')
    alpha, beta = alignment_constant(pose_seqs[0, 0], ref_pose)

    pose_seqs = pose_seqs * alpha + beta[None, None, :, None]

    # important not to let return value be gc'd (anims won't run otherwise!)
    anim = draw_poses(
        'Completed poses in %s' % args.completion_path,
        d['parents'],
        pose_seqs,
        frame_paths=[frame_paths] * 3,
        subplot_titles=seq_names,
        fps=50 / 9.0,
        crossover=d['crossover_time'])
    if args.vid_dir is not None:
        # save video
        print('Saving video')
        try:
            os.makedirs(args.vid_dir)
        except FileExistsError:
            pass

        bn = os.path.basename(args.completion_path).rsplit('.')[0]
        key = d['vid_name'] + '-' + bn
        anim.save(os.path.join(args.vid_dir, key + '.mp4'),
                  writer='avconv',
                  # no idea what bitrate defaults to, but empircally it seems
                  # to be around 1000 (?)
                  bitrate=3000,
                  # dpi defaults to 300
                  dpi=300,
                  fps=50/3.0)
    else:
        print('Showing sequence')
        plt.show()
