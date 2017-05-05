#!/usr/bin/env python3
"""Calculates validation statistics from a file of true/predicted poses (and/or
skeletons)."""

from argparse import ArgumentParser
import json
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


def pck(true_poses, pred_poses, valid_mask, joints, scales, thresholds=[]):
    """Expected PCK for listed keypoints at various thresholds."""
    assert true_poses.shape == pred_poses.shape, \
        "prediction and gt shape should match"
    assert true_poses.shape[-2] == 2, "need xy axis as second last one"

    dists = np.linalg.norm(true_poses - pred_poses, axis=-2)
    dists = dists[..., joints]

    valid_mask = valid_mask.astype('bool')
    assert valid_mask.shape == dists.shape[:2]

    rv = []
    for threshold in thresholds:
        under_thresh = dists < (scales[..., None] * threshold)
        sums = np.sum(under_thresh & valid_mask[..., None], axis=0).sum(axis=1)
        # this might yield NaNs, but that's actually okay, because I'm not sure
        # that there's a better way to signal "this result is garbage and I
        # can't get a better one"
        accs = sums / (len(joints) * np.sum(valid_mask, axis=0))
        rv.append(accs)

    return rv


def broadcast_out_preds(preds, *arrs):
    # Predictions are N*S*T*2*J, but truep poses are N*T*2*J, and various other
    # sturctures are N*T*... instead of N*S*T*... This function matches them
    # all together by inserting a new second axis and tiling along it S times.
    # It then flattens the result for easier stats calculation.
    reps = preds.shape[1]
    fdim = reps * preds.shape[0]
    return_arrays = [preds.reshape((fdim, ) + preds.shape[2:])]
    for arr in arrs:
        assert arr.shape[0] == preds.shape[0]
        assert arr.shape[1] == preds.shape[2]
        arr_rep = np.repeat(arr[:, None, ...], reps, axis=1)
        return_arrays.append(arr_rep.reshape((fdim, ) + arr_rep.shape[2:]))
    return return_arrays


def to_str(data):
    if isinstance(data, bytes):
        return data.decode('utf8')
    elif isinstance(data, str):
        return data
    raise TypeError("Don't know how to convert %s to string" % type(data))


parser = ArgumentParser()
parser.add_argument('h5_file', help='input stats file to read from')
parser.add_argument(
    '--output_dir', default='stats', help='where to put output .csv file')

if __name__ == '__main__':
    args = parser.parse_args()

    with h5py.File(args.h5_file) as fp:
        method = to_str(fp['/method_name'].value)
        extra_data = json.loads(to_str(fp['/extra_data'].value))

        has_2d = 'parents_2d' in fp
        if has_2d:
            pck_joints = extra_data['pck_joints']

            poses_2d_true = fp['/poses_2d_true'].value
            poses_2d_pred = fp['/poses_2d_pred'].value
            assert poses_2d_true.ndim == 4
            assert poses_2d_pred.ndim == 5
            assert poses_2d_pred.shape[:1] + poses_2d_pred.shape[
                2:] == poses_2d_true.shape
            scales = fp['/scales_2d'].value
            if '/is_usable' in fp:
                valid_mask = fp['/is_usable'].value
            else:
                valid_mask = np.ones_like(scales, dtype=bool)
            assert valid_mask.ndim == 2
            assert valid_mask.shape == poses_2d_true.shape[:2]
            assert valid_mask.shape == scales.shape

            flat_2d_pred, flat_2d_true, flat_scales, flat_mask \
                = broadcast_out_preds(poses_2d_pred, poses_2d_true, scales,
                                      valid_mask)

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
        # thresholds are expressed in "fraction of distance across bounding
        # box", so only really low thresholds are relevant
        thresholds = np.linspace(0.0001, 0.015, 100)
        for group_name, joints in pck_joints.items():
            group_pcks = pck(flat_2d_true, flat_2d_pred, flat_mask, joints,
                             flat_scales, thresholds)
            # each of "group_pcks" is a length-T vector
            pck_table = np.stack(group_pcks, axis=1)
            # add times to beginning
            pck_table = np.concatenate(
                [np.arange(len(pck_table))[:, None], pck_table], axis=1)
            dest_path = os.path.join(dest_dir,
                                     'pck_%s_%s.csv' % (method, group_name))
            print('Writing stats for "%s" to %s' % (group_name, dest_path))
            header = 'Time,' + ','.join('@%g' % t for t in thresholds)
            np.savetxt(
                dest_path, pck_table, delimiter=',', fmt='%f', header=header)

    if has_3d:
        # calculate expected JAE
        angle_errors = []
        for s in range(skeletons_3d_pred.shape[1]):
            err = angle_error(skeletons_3d_true, skeletons_3d_pred[:, s],
                              parents_3d)
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
