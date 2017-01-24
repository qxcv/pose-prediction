#!/usr/bin/env python3

"""Conversion script for Penn Action dataset."""

from argparse import ArgumentParser
from glob import glob
from h5py import File
import numpy as np
from os import path
from scipy.io import loadmat
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
    assert 20 < abs(scale) < 800, scale
    joints /= scale

    # Compute actual data (relative offsets are easier to learn)
    relpose = np.zeros_like(joints)
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
        action_ids = {}
        for mat_path in tqdm(file_list):
            id_str, relpose, action = load_seq(mat_path)
            prefix = '/seqs/' + id_str + '/'
            fp[prefix + 'poses'] = relpose
            fp[prefix + 'action'] = action
        fp['/parents'] = np.array(PARENTS, dtype=int)