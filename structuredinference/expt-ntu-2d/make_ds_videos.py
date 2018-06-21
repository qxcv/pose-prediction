#!/usr/bin/env python3
"""Turn original dataset sequences into videos. Super duper cut-and-pasted."""

import argparse
import json
import os
import random

import h5py
import numpy as np
import matplotlib.pyplot as plt

import addpaths  # noqa
from plot_seqs import draw_poses
from video_common import ZipVideo, parse_name

H5_PATH = './ntu_data.h5'
ZIPS_DIR = '/data/home/sam/ntu-rgbd/'
ZIPS_DIR = '/home/sam/sshfs/paloalto' + ZIPS_DIR  # XXX

parser = argparse.ArgumentParser()
parser.add_argument(
    '--vid_name', help='name of video sequence to show (otherwise random)')
parser.add_argument(
    '--save',
    default=None,
    metavar='DEST',
    help='if supplied, save to video file instead of showing')


if __name__ == '__main__':
    args = parser.parse_args()
    vid_name = args.vid_name

    with h5py.File(H5_PATH) as fp:
        if vid_name is None:
            vid_name = random.choice(list(fp['/seqs']))
            print("Selected video '%s'" % vid_name)
        poses = fp['/seqs/' + vid_name + '/poses'].value
        action_ids = fp['/seqs/' + vid_name + '/actions'].value
        action_name_map = np.asarray(
            json.loads(fp['/action_names'].value.tostring().decode('utf8')))
        parents = fp['/parents'].value

    # read out all the frames we need
    zip_name, avi_name, start_frame, end_frame = parse_name(vid_name)
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    video = ZipVideo(zip_path, avi_name)
    frames = [
        video.get_frame(frame_idx)
        for frame_idx in range(start_frame, end_frame + 1)
    ]

    pose_seqs = poses[None, ...]
    action_labels = action_name_map[action_ids]
    seq_names = ['True poses']

    assert len(action_labels) == len(poses)
    print('Action names: ' + ', '.join(action_name_map))
    print('Actions: ' + ', '.join(action_labels))

    # important not to let return value be gc'd (anims won't run otherwise!)
    anims = draw_poses(
        'Poses in %s' % vid_name,
        parents,
        pose_seqs,
        frames=[frames],
        subplot_titles=seq_names,
        # always plot at 10fps so that we can actually see the action :P
        fps=10,
        action_labels=action_labels)
    if args.save is not None:
        print('Saving video to %s' % args.save)
        anims.save(args.save)
    else:
        plt.show()
