#!/usr/bin/env python3
"""Constant-velocity and zero-velocity baselines for 2D and 3D pose estimation
datasets."""

import sys
sys.path.append('keras')

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from p2d_loader import P2DDataset, P3DDataset  # noqa: E402
from common import mkdir_p  # noqa: E402

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
    # TODO: is this the right way to do it? does velocity over entire
    # conditioning sequence make more sense?
    velocities = f32(input_poses[:, -1:] - input_poses[:, -2:-1])
    nsteps = np.arange(steps_to_predict).reshape((1, -1, 1, 1)) + 1
    nsteps = f32(nsteps)
    # velocities * nsteps is broadcasting abuse :-)
    rv = f32(velocities * nsteps + input_poses[:, -1:])
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


def write_baseline(out_prefix, cond_on, pred_on, parents, is_3d, usable,
                   scales, extra_data, method, cond_actions, pred_actions,
                   action_names):
    meth_name = method.__name__
    steps_to_predict = pred_on.shape[1]
    out_path = out_prefix + '_' + meth_name + '.h5'
    print('Writing %s baseline to %s' % (meth_name, out_path))
    if not is_3d:
        # in 2D, XY is stored second, while in 3D, XYZ is stored last (yes this
        # is a mess, but it takes time to fix)
        cond_on = cond_on.transpose((0, 1, 3, 2))
    result = method(cond_on, steps_to_predict)[:, None, ...]
    if not is_3d:
        del cond_on
        # move back into N,T,XY,J format
        result = result.transpose((0, 1, 2, 4, 3))
        assert (result.shape[0],) + result.shape[2:] == pred_on.shape, \
            (result.shape, pred_on.shape)
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
                data=result)
        else:
            fp['/parents_2d'] = parents
            fp.create_dataset(
                '/poses_2d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on)
            fp['/scales_2d'] = f32(scales)
            fp.create_dataset(
                '/poses_2d_pred',
                compression='gzip',
                shuffle=True,
                data=result)
            # also action data
            fp.create_dataset(
                '/cond_actions_2d', compression='gzip', data=cond_actions)
            fp.create_dataset(
                '/pred_actions_2d', compression='gzip', data=pred_actions)
            fp['/action_names'] = json.dumps(action_names)
        fp['/is_usable'] = usable
        fp['/extra_data'] = json.dumps(extra_data)


if __name__ == '__main__':
    args = parser.parse_args()
    extra_data = {}

    if args.is_3d:
        dataset = P3DDataset(args.dataset_path)
        cond_on, pred_on = dataset.get_ds_for_eval(train=False)
        cond_on_orig = f32(dataset.reconstruct_skeletons(cond_on))
        pred_on_orig = f32(dataset.reconstruct_skeletons(pred_on))
        pred_val = pred_scales = None
        cond_actions = pred_actions = action_names = None
    else:
        dataset = P2DDataset(args.dataset_path, 32)
        evds = dataset.get_ds_for_eval(train=False)
        cond_on = evds['conditioning']
        pred_on = evds['prediction']
        pred_scales = evds['prediction_scales']
        if dataset.has_sparse_annos:
            pred_val = evds['prediction_valids']
        else:
            pred_val = None
        extra_data['pck_joints'] = dataset.pck_joints
        seq_ids = evds['seq_ids']
        pred_frame_numbers = evds['prediction_frame_nums']
        cond_frame_numbers = evds['conditioning_frame_nums']
        cond_on_orig = f32(
            dataset.reconstruct_poses(cond_on, seq_ids, cond_frame_numbers))
        pred_on_orig = f32(
            dataset.reconstruct_poses(pred_on, seq_ids, pred_frame_numbers))
        pred_actions = evds['prediction_actions']
        cond_actions = evds['conditioning_actions']
        action_names = dataset.action_names

    mkdir_p(os.path.dirname(args.output_prefix))

    if pred_val is None:
        pred_val = np.ones(pred_on_orig.shape[:2], dtype=bool)

    if pred_scales is None:
        pred_scales = np.ones(pred_on_orig.shape[:2], dtype='float32')

    write_baseline(args.output_prefix, cond_on_orig, pred_on_orig,
                   dataset.parents, args.is_3d, pred_val, pred_scales,
                   extra_data, zero_velocity, cond_actions, pred_actions,
                   action_names)
    write_baseline(args.output_prefix, cond_on_orig, pred_on_orig,
                   dataset.parents, args.is_3d, pred_val, pred_scales,
                   extra_data, constant_velocity, cond_actions, pred_actions,
                   action_names)
