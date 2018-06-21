#!/usr/bin/env python3
"""Turn completion files into actual SxS videos of the completion, the ground
truth, etc. Probably shares a lot with the other completion video scripts."""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import addpaths  # noqa
from plot_2d_seqs import draw_poses
from common_pp.completion_video_common import load_sorted_paths

FRAME_DIR = '/data/home/cherian/IkeaDataset/Frames/'
DB_PATH = '/data/home/cherian/IkeaDataset/IkeaClipsDB_withactions.mat'
POSE_DIR = '/home/sam/etc/cpm-keras/ikea-mat-poses/'

parser = argparse.ArgumentParser()
parser.add_argument('completion_path', help='path to .json completion file')
parser.add_argument(
    '--vid-dir',
    type=str,
    default=None,
    help='save videos to this directory instead of showing poses')

if __name__ == '__main__':
    args = parser.parse_args()

    db = loadmat(DB_PATH, squeeze_me=True)['IkeaDB']
    # could get just one entry (one relevant to our vid) instead of looping
    # over all. oh well
    meta_dict = {}
    for video_entry in db:
        clip_path = video_entry['clip_path']
        prefix = '/data/home/cherian/IkeaDataset/Frames/'
        assert clip_path.startswith(prefix)
        path_suffix = clip_path[len(prefix):]
        # This is same number used to identify pose clip (not sequential!)
        tmp2_id = video_entry['video_id']
        new_name = 'vid%d' % tmp2_id
        meta_dict[new_name] = {'path_suffix': path_suffix, 'tmp2_id': tmp2_id}

    with open(args.completion_path) as fp:
        d = json.load(fp)
    vid_name = d['vid_name']
    meta = meta_dict[vid_name]
    path_suffix = meta['path_suffix']
    tmp2_id = meta['tmp2_id']
    # tmp2_id = int(re.match(r'^vid(\d+)$', vid_name).groups()[0])
    all_frame_fns = load_sorted_paths(os.path.join(FRAME_DIR, path_suffix))
    # for some reason there is one video directory with a subdirectory that has
    # a numeric name
    all_frame_fns = [f for f in all_frame_fns if f.endswith('.jpg')]
    frame_paths = [all_frame_fns[i] for i in d['frame_inds']]

    pose_seqs = np.stack(
        (d['true_poses'], d['prior_poses'], d['posterior_poses']), axis=0)
    seq_names = ['True poses', 'Prior prediction', 'Posterior prediction']

    pose_mat_path = os.path.join(POSE_DIR, 'pose_clip_%d.mat' % tmp2_id)
    pose_mat = loadmat(pose_mat_path, squeeze_me=True)

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
