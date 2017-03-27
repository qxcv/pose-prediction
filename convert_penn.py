#!/usr/bin/env python3

"""Conversion script for Penn Action dataset."""

from argparse import ArgumentParser
from glob import glob
from h5py import File
from json import dumps
import numpy as np
from os import path
from scipy.io import loadmat
import sys
from textwrap import wrap
from tqdm import tqdm

# Joint indices from README:
# 1.  head
# 2.  left_shoulder  3.  right_shoulder
# 4.  left_elbow     5.  right_elbow
# 6.  left_wrist     7.  right_wrist
# 8.  left_hip       9.  right_hip
# 10. left_knee      11. right_knee
# 12. left_ankle     13. right_ankle

#      head0, lsho1, rsho2, lelb3, relb4, lwri5, rwri6, lhi7, rhi8, lk9, rk10
PARENTS = [0,     0,     0,     1,     2,     3,     4,    1,    2,   7,    8,
#       lk11, rk12  # noqa
           9,   10]
ACTION_NAMES = [
    'baseball_pitch', 'baseball_swing', 'bench_press', 'bowl',
    'clean_and_jerk', 'golf_swing', 'jump_rope', 'jumping_jacks', 'pullup',
    'pushup', 'situp', 'squat', 'strum_guitar', 'tennis_forehand',
    'tennis_serve'
]

parser = ArgumentParser()
parser.add_argument('penn_path', help='path to Penn Action dataset')
parser.add_argument('dest', help='path for HDF5 output file')


def load_seq(mat_path):
    # id_str will be left-zero-padded
    id_str, _ = path.basename(mat_path).rsplit('.mat', 1)

    mat = loadmat(mat_path, squeeze_me=True)

    # 'x', 'y' are T*J arrays. I want to convert to T*J*XY
    x, y = mat['x'], mat['y']
    joints = np.stack([x, y], axis=-1)
    # originally the coords were uint16, which caused major problems when
    # normalising
    joints = np.array(joints, dtype=float)

    visible = mat['visibility'].astype(bool)
    # 'visible' is T*J, we need it to be T*2*J
    actual_visible = np.tile(visible[..., np.newaxis], (1, 1, 2))
    assert np.all(actual_visible[:, :, 0] == visible)
    assert actual_visible.shape == joints.shape, \
        'expected %s, got %s' % (joints.shape, actual_visible.shape)

    # Normalise by median hip-shoulder distance (will invisible joints work?)
    # left shoulder/right hip
    valid_lr = visible[:, 1] & visible[:, 8]
    hsd_lr = np.linalg.norm(joints[:, 1] - joints[:, 8], axis=1)
    hsd_lr = hsd_lr[valid_lr]
    # right shoulder/left hip
    valid_rl = visible[:, 2] & visible[:, 7]
    hsd_rl = np.linalg.norm(joints[:, 2] - joints[:, 7], axis=1)
    hsd_rl = hsd_rl[valid_rl]
    hsd_cat = np.concatenate((hsd_lr, hsd_rl))
    if hsd_cat.size == 0:
        # can't actually calculate a scale
        return None
    scale = np.median(hsd_cat)
    # Make sure that scale is sane
    if abs(scale) < 40 or abs(scale) > 400:
        return None

    # Need to be T*XY*J
    joints = joints.transpose((0, 2, 1))
    assert joints.shape[1] == 2, joints.shape
    assert joints.shape[0] == x.shape[0], joints.shape

    return id_str, joints, mat['action'], actual_visible, scale


if __name__ == '__main__':
    args = parser.parse_args()
    file_list = glob(path.join(args.penn_path, 'labels', '*.mat'))
    with File(args.dest, 'w') as fp:
        skipped = []
        for mat_path in tqdm(file_list):
            rv = load_seq(mat_path)
            if rv is None:
                skipped.append(mat_path)
                continue
            id_str, joints, action, visible, scale = rv
            action_id = ACTION_NAMES.index(action)
            prefix = '/seqs/' + id_str + '/'
            fp[prefix + 'poses'] = joints
            fp[prefix + 'actions'] = np.full((len(joints),), action_id)
            fp[prefix + 'valid'] = visible
            fp[prefix + 'scale'] = scale
        fp['/parents'] = np.array(PARENTS, dtype=int)
        fp['/action_names'] = np.array([ord(c) for c in dumps(ACTION_NAMES)],
                                       dtype='uint8')
        fp['/num_actions'] = len(ACTION_NAMES)
        if skipped:
            print('WARNING: skipped %i seq(s) due to scale:' % len(skipped),
                  file=sys.stderr)
            print('\n'.join(wrap(', '.join(skipped))))
