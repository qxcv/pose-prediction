#!/usr/bin/env python3
"""Calculates validation statistics from a file of true/predicted poses (and/or
skeletons)."""

from argparse import ArgumentParser
import os

import numpy as np
import h5py

from expmap import toposort, exps_to_eulers


def angle_error(true_exp_skels, pred_exp_skels, parents):
    """L2 diference in Euler angle parameterisations, as in S-RNN paper."""
    # assumes that we have shape N*T*J*3
    assert true_exp_skels.shape == pred_exp_skels.shape, \
        "gt and prediction shapes must match"
    assert true_exp_skels.ndim == 4
    N, T, J, D = true_exp_skels.shape
    assert D == 3, "Need 3D data"

    # root contains offset from previous frame rather than parent-relative
    # angle, so ignore it (i.e. exclude zeroth element of toposorted joints)
    valid_joints = toposort(parents)[1:]
    true_euler = exps_to_eulers(true_exp_skels[..., valid_joints, :])
    pred_euler = exps_to_eulers(pred_exp_skels[..., valid_joints, :])

    dists = np.linalg.norm(true_euler - pred_euler, axis=-1)

    # average over both joints and sequence number, returning length-T array
    rv = np.mean(dists, axis=0).mean(axis=1)
    assert rv.ndim == 1 and rv.size == true_exp_skels.shape[1]

    return rv


def pck(true_poses, pred_poses, thresholds=[]):
    """Expected PCK for *every* keypoint at various thresholds."""
    assert true_poses == pred_poses, "prediction and gt shape should match"
    assert true_poses.shape[-1] == 2, "need xy axis as last one"

    dists = np.linalg.norm(true_poses - pred_poses, axis=-1)
    rv = []

    for threshold in thresholds:
        accs = np.mean(dists < threshold, axis=0)
        rv.append(accs)

    return rv


parser = ArgumentParser()
parser.add_argument('h5_file', help='input stats file to read from')
parser.add_argument(
    '--output_dir', default='stats', help='where to put output .csv file')

if __name__ == '__main__':
    args = parser.parse_args()

    with h5py.File(args.h5_file) as fp:
        method = fp['/method_name'].value
        assert isinstance(method, str)

        has_2d = 'parents_2d' in fp
        if has_2d:
            pck_thresholds = fp['/pck_thresholds'].value
            poses_2d_true = fp['/poses_2d_true'].value
            poses_2d_pred = fp['/poses_2d_pred'].value
            assert poses_2d_true.ndim == 4
            assert poses_2d_pred.ndim == 5
            assert poses_2d_pred.shape[:1] + poses_2d_pred.shape[
                2:] == poses_2d_true.shape

        has_3d = 'parents_3d' in fp
        if has_3d:
            parents_3d = fp['/parents_3d'].value
            skeletons_3d_true = fp['/skeletons_3d_true'].value
            skeletons_3d_pred = fp['/skeletons_3d_pred'].value
            assert skeletons_3d_true.ndim == 4
            assert skeletons_3d_pred.ndim == 5
            assert skeletons_3d_pred.shape[:1] + skeletons_3d_pred.shape[
                2:] == skeletons_3d_true.shape

    print('2D stats collected? %s' % has_2d)
    print('3D stats collected? %s' % has_3d)

    dest_dir = args.output_dir
    try:
        os.makedirs(dest_dir)
    except FileExistsError:
        pass

    if has_2d:
        # calculate expected PCK at various thresholds
        pck_samples = []
        for s in range(poses_2d_pred.shape[1]):
            pck(
                poses_2d_true, )
        pcks = np.mean(pck_samples, axis=0)
        pass

    if has_3d:
        # calculate expected JAE
        angle_errors = []
        for s in range(skeletons_3d_pred.shape[1]):
            err = angle_error(skeletons_3d_true, skeletons_3d_pred[:, s], parents_3d)
            angle_errors.append(err)
        # average over samples
        mjae = np.mean(angle_errors, axis=0)
        assert mjae.ndim == 1, \
            "Expected length T array, got array shape " + str(mjae.shape)

        # add first column for time step and write to file
        mjae_w_inds = np.stack([np.arange(len(mjae)), mjae], axis=1)
        out_path = os.path.join(dest_dir, 'mjae_%s.csv' % method)
        np.savetxt(
            out_path, mjae_w_inds, delimiter=',', fmt='%f', header='Time,MJAE')
