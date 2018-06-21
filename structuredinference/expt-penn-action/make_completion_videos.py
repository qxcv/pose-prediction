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
from common_pp.completion_video_common import load_sorted_paths, alignment_constant

# recall frame layout: frames/ has subdirectories with 4-digit video IDs, which
# themselves have .jpg files with six-digit frame numbers.
FRAME_DIR = '/data/home/sam/Penn_Action/frames/'
FRAME_DIR = '/home/sam/sshfs/paloalto' + FRAME_DIR  # XXX
# pose directory is similar, but with no nesting: 4-digit IDs with .mat on the
# end give you poses (with "x" and "y" keys mapping to T*13 arrays of
# positions)
POSE_DIR = '/data/home/sam/Penn_Action/labels/'
POSE_DIR = '/home/sam/sshfs/paloalto' + POSE_DIR # XXX

parser = argparse.ArgumentParser()
parser.add_argument('completion_path', help='path to .json completion file')

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

    pose_mat_path = path.join(POSE_DIR, d['vid_name'] + '.mat')
    pose_mat = loadmat(pose_mat_path, squeeze_me=True)
    ref_pose = np.stack((pose_mat['x'][0], pose_mat['y'][0]), axis=0)
    alpha, beta = alignment_constant(pose_seqs[0, 0], ref_pose)

    pose_seqs = pose_seqs * alpha + beta[None, None, :, None]

    # important not to let return value be gc'd (anims won't run otherwise!)
    anims = draw_poses(
        'Completed poses in %s' % args.completion_path,
        d['parents'],
        pose_seqs,
        frame_paths=[frame_paths] * 3,
        subplot_titles=seq_names,
        fps=50 / 9.0,
        crossover=d['crossover_time'])
    plt.show()
