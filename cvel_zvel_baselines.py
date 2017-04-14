#!/usr/bin/env python3
"""Constant-velocity and zero-velocity baselines for 2D and 3D pose estimation
datasets."""

import sys
sys.path.append('keras')

import argparse
import os

import h5py
import numpy as np

from p2d_loader import P2DDataset, P3DDataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', help='path to input HDF5 file')
parser.add_argument(
    'output_prefix',
    help='prefix for output files (will get _zvel.h5 and _cvel.h5 appended)')
parser.add_argument(
    '--3d',
    action='store_true',
    dest='is_3d',
    default=False,
    help='treat this as a 3D dataset')


def f32(x):
    return np.asarray(x, dtype='float32')


def constant_velocity(input_poses, steps_to_predict):
    N, T, J, D = input_poses.shape
    assert J > D, \
        "hmm, should have XY or XYZ *last*. Is this the case?"
    # velocities for each using only last two frames
    # TODO: is this the right way to do it? does velocity over entire sequence
    # make more sense?
    velocities = f32(input_poses[:, -1:] - input_poses[:, -2:-1])
    nsteps = np.arange(steps_to_predict).reshape((1, -1, 1, 1)) + 1
    nsteps = f32(nsteps)
    # broadcasting abuse :-)
    rv = f32(velocities * nsteps)
    assert rv.shape == (N, steps_to_predict, J, D)
    return rv


def zero_velocity(input_poses, steps_to_predict):
    """Just copy the last pose over and over."""
    N, T, J, D = input_poses.shape
    assert J > D, \
        "should have XY or XYZ *last*"
    last_pose = f32(input_poses[:, -1:])
    rv = f32(np.tile(last_pose, (1, steps_to_predict, 1, 1)))
    assert rv.shape == (N, steps_to_predict, J, D)
    return rv


def write_baseline(out_prefix, cond_on, pred_on, parents, is_3d, method):
    meth_name = method.__name__
    steps_to_predict = pred_on.shape[1]
    out_path = out_prefix + '_' + meth_name + '.h5'
    print('Writing %s baseline to %s' % (meth_name, out_path))
    with h5py.File(out_path, 'w') as fp:
        fp['/method_name'] = meth_name
        if is_3d:
            fp['/parents_3d'] = parents
            fp.create_dataset(
                '/skeletons_3d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on)
            fp.create_dataset(
                '/skeletons_3d_pred',
                compression='gzip',
                shuffle=True,
                data=method(cond_on, steps_to_predict)[:, None, ...])
        else:
            fp['/parents_2d'] = parents
            fp.create_dataset(
                '/poses_3d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on)
            fp.create_dataset(
                '/poses_3d_pred',
                compression='gzip',
                shuffle=True,
                data=method(cond_on, steps_to_predict)[:, None, ...])


if __name__ == '__main__':
    args = parser.parse_args()

    if args.is_3d:
        dataset = P3DDataset(args.dataset_path)
        cond_on, pred_on = dataset.get_ds_for_eval(train=False)
        cond_on_orig = f32(dataset.reconstruct_skeletons(cond_on))
        pred_on_orig = f32(dataset.reconstruct_skeletons(pred_on))
    else:
        dataset = P2DDataset(args.dataset_path)
        cond_on, pred_on = dataset.get_ds_for_eval(train=False)
        cond_on_orig = f32(dataset.reconstruct_poses(cond_on))
        pred_on_orig = f32(dataset.reconstruct_poses(pred_on))

    try:
        os.makedirs(os.path.dirname(args.output_prefix))
    except FileExistsError:
        pass

    write_baseline(args.output_prefix, cond_on_orig, pred_on_orig,
                   dataset.parents, args.is_3d, constant_velocity)
    write_baseline(args.output_prefix, cond_on_orig, pred_on_orig,
                   dataset.parents, args.is_3d, zero_velocity)
