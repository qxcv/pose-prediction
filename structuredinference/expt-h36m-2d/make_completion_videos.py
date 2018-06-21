#!/usr/bin/env python2
"""Turn completion files into actual SxS videos of the completion, the ground
truth, etc. Probably shares a lot with the other completion video scripts."""

import argparse
import json
import os
from os import path

import numpy as np
import matplotlib.pyplot as plt
import cv2

import addpaths  # noqa
from plot_2d_seqs import draw_poses

H36M_DIR = '/data/home/sam/h3.6m/'
H36M_DIR = '/home/sam/sshfs/paloalto' + H36M_DIR  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('completion_path', help='path to .json completion file')
parser.add_argument(
    '--vid-dir',
    type=str,
    default=None,
    help='save videos to this directory instead of showing poses')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.completion_path) as fp:
        d = json.load(fp)

    vid_name = d['vid_name']
    _, sub_num_s, act_name, cam_id = vid_name.split(':')
    full_ident = '%s.%s' % (act_name, cam_id)
    subj_dir = path.join(H36M_DIR, 'S' + sub_num_s)
    vid_path = path.join(subj_dir, 'Videos', full_ident + '.mp4')

    cap = cv2.VideoCapture(vid_path)
    assert cap.isOpened(), "Failed to open '%s' (OpenCV doesn't give any " \
        "error message beyond the implicit \"fuck you\" of failure)"

    # to get a frame:
    n_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_seq = []
    for fr_ind in d['frame_inds']:
        assert 0 <= fr_ind < n_frames, 'frame %d out of range [0, %d)' \
            % (fr_ind, n_frames)
        assert cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, fr_ind), \
            "could not skip to frame %d" % fr_ind
        succ, frame = cap.read()
        assert succ, "frame-reading failed on frame %d" % fr_ind
        # OpenCV has weird BGR loading convention; need RGB instead
        rgb_frame = frame[:, :, ::-1]
        frame_seq.append(rgb_frame)

    pose_seqs = np.stack(
        (d['true_poses'], d['prior_poses'], d['posterior_poses']), axis=0)
    seq_names = ['True poses', 'Prior prediction', 'Posterior prediction']

    # important not to let return value be gc'd (anims won't run otherwise!)
    anim = draw_poses(
        'Completed poses in %s' % args.completion_path,
        d['parents'],
        pose_seqs,
        frames=[frame_seq] * pose_seqs.shape[0],
        subplot_titles=seq_names,
        fps=50 / 9.0,
        crossover=d['crossover_time'])
    if args.vid_dir is not None:
        # save video
        print('Saving video')
        try:
            os.makedirs(args.vid_dir)
        except OSError:
            pass

        bn = os.path.basename(args.completion_path).rsplit('.')[0]
        key = d['vid_name'] + '-' + bn
        anim.save(
            os.path.join(args.vid_dir, key + '.mp4'),
            writer='avconv',
            # no idea what bitrate defaults to, but empircally it seems
            # to be around 1000 (?)
            bitrate=3000,
            # dpi defaults to 300
            dpi=300,
            fps=50 / 3.0)
    else:
        print('Showing sequence')
        plt.show()
