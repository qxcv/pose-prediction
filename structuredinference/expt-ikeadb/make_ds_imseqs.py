#!/usr/bin/env python2
"""Plot some 2D poses as a single, static sequence with Mayavi. The actual
plotting work goes on in plot_2d_seq_stat (pose-prediction/keras), though."""

import argparse
import json
import os
import re

import h5py

import numpy as np

from scipy.io import loadmat

import addpaths  # noqa
from plot_2d_seq_static import draw_poses
from common_pp.completion_video_common import load_sorted_paths

H5_PATH = './ikea_action_data.h5'
DB_PATH = '/data/home/cherian/IkeaDataset/IkeaClipsDB_withactions.mat'
DB_PATH = '/home/sam/sshfs/paloalto' + DB_PATH  # XXX
FRAME_DIR = '/data/home/cherian/IkeaDataset/Frames/'
FRAME_DIR = '/home/sam/sshfs/paloalto' + FRAME_DIR  # XXX
POSE_DIR = '/home/sam/sshfs/paloalto/etc/cpm-keras/ikea-mat-poses/'  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('vid_name', help='seq ident in HDF5 file (e.g. vid103)')

if __name__ == '__main__':
    args = parser.parse_args()
    vid_name = args.vid_name

    with h5py.File(H5_PATH) as fp:
        action_names = np.asarray(json.loads(fp['/action_names']
                                  .value
                                  .tobytes()
                                  .decode('utf8')))
        parents = fp['/parents'].value
        poses = fp['/seqs/' + vid_name + '/poses'].value
        actions = fp['/seqs/' + vid_name + '/actions'].value

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

    meta = meta_dict[vid_name]
    path_suffix = meta['path_suffix']
    tmp2_id = meta['tmp2_id']
    tmp2_id = int(re.match(r'^vid(\d+)$', vid_name).groups()[0])
    all_frame_fns = load_sorted_paths(os.path.join(FRAME_DIR, path_suffix))
    # for some reason there is one video directory with a subdirectory that has
    # a numeric name
    frame_paths = [f for f in all_frame_fns if f.endswith('.jpg')]

    action_labels = action_names[actions]
    assert len(action_labels) == len(poses)
    print('Action names: ' + ', '.join(action_names))
    print('Actions: ' + ', '.join(action_labels))

    fig = draw_poses(
        'Original training poses in %s' % H5_PATH,
        parents,
        poses,
        frame_paths=frame_paths,
        # action_labels=action_labels
        fps=50)
