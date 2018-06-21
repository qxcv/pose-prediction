#!/usr/bin/env python3
"""Turn evaluation file into sequences of videos. Not sure how I'm going to
make this method-agnostic."""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import h5py

import addpaths  # noqa
from plot_seqs import draw_poses
from video_common import ZipVideo, parse_name

H5_PATH = './ntu_data.h5'
ZIPS_DIR = '/data/home/sam/ntu-rgbd/'
ZIPS_DIR = '/home/sam/sshfs/paloalto' + ZIPS_DIR  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('results_h5_path', help='path to .h5 from eval code')
parser.add_argument(
    '--vid-dir',
    type=str,
    default=None,
    help='save videos to this directory instead of showing poses')
parser.add_argument(
    '--best-only',
    action='store_true',
    default=False,
    help='only show best sample')


def force_string(maybe_str):
    if isinstance(maybe_str, str):
        return maybe_str
    elif isinstance(maybe_str, bytes):
        return maybe_str.decode('utf8')
    raise TypeError('no idea how to convert a %s' % type(maybe_str))


if __name__ == '__main__':
    args = parser.parse_args()

    ###########################################################################
    # v XXX: keep this generic, so I can factor out eventually
    with h5py.File(args.results_h5_path, 'r') as fp:
        num_seqs = len(fp['/poses_2d_pred'])
        # pick a random one!
        sel_idx = np.random.randint(num_seqs)
        json_vid_names = json.loads(force_string(fp['/seq_ids_2d_json'].value))
        vid_name = json_vid_names[sel_idx]
        print('Selected sample set %d (video name %s)' % (sel_idx, vid_name))
        seq_frame_inds_pred = fp['/pred_frame_numbers_2d'].value[sel_idx]
        true_poses_pred = fp['/poses_2d_true'].value[sel_idx]
        pred_poses_pred = fp['/poses_2d_pred'].value[sel_idx]
        seq_frame_inds_cond = fp['/cond_frame_numbers_2d'].value[sel_idx]
        true_poses_cond = fp['/poses_2d_cond_true'].value[sel_idx]
        pred_poses_cond = fp['/poses_2d_cond_pred'].value[sel_idx]
        cond_steps = true_poses_cond.shape[0]
        seq_frame_inds = np.concatenate(
            [seq_frame_inds_cond, seq_frame_inds_pred])
        true_poses = np.concatenate([true_poses_cond, true_poses_pred], axis=0)
        out_poses = np.concatenate([pred_poses_cond, pred_poses_pred], axis=1)
        if args.best_only:
            # only use the samples which are closest to the ground truth
            diffs = true_poses[None, ...] - out_poses
            # gives us N*S array; need to min over S
            sq_diffs = (diffs**2).sum(axis=-1).sum(axis=-1).T
            best_inds = np.argmin(sq_diffs, axis=1)
            ax1_lin = np.arange(out_poses.shape[1])
            out_poses = out_poses[None, best_inds, ax1_lin]
        num_samples = out_poses.shape[0]
        if args.best_only:
            seq_names = ['Estimated pose', ('Decoded pose', 'Forecasted pose')]
        else:
            seq_names = ['True poses'] + \
                        ['Sample %d' % d for d in range(num_samples)]
        pose_seqs = np.stack([true_poses] + [r for r in out_poses], axis=0)
        parents = fp['/parents_2d'].value
    # ^ XXX: keep this generic for refactoring purposes
    ###########################################################################

    print('vid_name:', vid_name)
    zip_name, avi_name, start_frame, end_frame = parse_name(vid_name)
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    video = ZipVideo(zip_path, avi_name)

    # we will preload frames
    orig_frames = [video.get_frame(frame_idx) for frame_idx in seq_frame_inds]

    ###########################################################################
    # v XXX: keep this generic for refactoring purposes
    # drop brightness the hacky way
    dark_final = (orig_frames[cond_steps - 1] / 2).astype('uint8')
    dark_list = [dark_final] * (len(orig_frames) - cond_steps)
    trunc_frames = orig_frames[:cond_steps] + dark_list

    # important not to let return value be gc'd (anims won't run otherwise!)
    anim = draw_poses(
        None,
        parents,
        pose_seqs,
        frames=[orig_frames] + [trunc_frames] * num_samples,
        subplot_titles=seq_names,
        fps=50 / 9.0,
        crossover=cond_steps)
    if args.vid_dir is not None:
        # save video
        print('Saving video')
        try:
            os.makedirs(args.vid_dir)
        except FileExistsError:
            pass

        key = vid_name + '-' + str(sel_idx)
        anim.save(
            os.path.join(args.vid_dir, key + '.mp4'),
            writer='avconv',
            # no idea what bitrate defaults to, but empirically it seems to be
            # around 1000 (?)
            bitrate=3000,
            # dpi defaults to 300
            dpi=300,
            fps=10)
    else:
        print('Showing sequence')
        plt.show()
    # ^ XXX: keep this generic for refactoring purposes
    ###########################################################################
