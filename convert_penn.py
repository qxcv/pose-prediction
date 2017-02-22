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

    # Normalise by median hip-shoulder distance (will invisible joints work?)
    # left shoulder/right hip
    hsd_lr = np.linalg.norm(joints[:, 1] - joints[:, 8], axis=1)
    # right shoulder/left hip
    hsd_rl = np.linalg.norm(joints[:, 2] - joints[:, 7], axis=1)
    scale = np.median(np.concatenate((hsd_lr, hsd_rl)))
    # Make sure that scale is sane
    if abs(scale) < 40 or abs(scale) > 400:
        return None
    joints /= scale

    # Compute actual data (relative offsets are easier to learn)
    relpose = np.zeros_like(joints, dtype='float32')
    relpose[0, 0, :] = 0
    # Head position records delta from previous frame
    relpose[1:, 0] = joints[1:, 0] - joints[:-1, 0]
    # Other joints record delta from parents
    for jt in range(1, len(PARENTS)):
        pa = PARENTS[jt]
        relpose[:, jt] = joints[:, jt] - joints[:, pa]

    return id_str, relpose, mat['action']


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
            id_str, relpose, action = rv
            action_id = ACTION_NAMES.index(action)
            prefix = '/seqs/' + id_str + '/'
            fp[prefix + 'poses'] = relpose
            fp[prefix + 'actions'] = np.full((len(relpose),), action_id)
        fp['/parents'] = np.array(PARENTS, dtype=int)
        fp['/action_names'] = np.array([chr(c) for c in dumps(ACTION_NAMES)],
                                       dtype='uint8')
        fp['/num_actions'] = len(ACTION_NAMES)
        if skipped:
            print('WARNING: skipped %i seq(s) due to scale:' % len(skipped),
                  file=sys.stderr)
            print('\n'.join(wrap(', '.join(skipped))))
