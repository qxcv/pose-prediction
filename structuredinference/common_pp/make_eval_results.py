#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Write out predictions file for this dataset and method. Output will be
processed by ``make_stats.py`` in ``pose-prediction`` repo."""

import addpaths  # noqa: F401

from argparse import ArgumentParser
import h5py
import json
import os
import shlex
import sys
import time

import numpy as np
from theano import config

from stinfmodel_fast import evaluate as DKF_evaluate
from stinfmodel_fast.dkf import DKF
from utils.misc import removeIfExists
import p2d_loader

# do this last because it's in the current dir
sys.path.append(os.getcwd())
from load import loadDataset  # noqa: E402

# v XXX
# # I want to get a collossal 2D vector of (mu, log_var) rows at the end of
# # this :)
# inferred_z_params = []
# rolled_z_params = []
# ^ XXX


def f32(x):
    return np.asarray(x, dtype='float32')


def fX(x):
    return np.asarray(x, dtype=config.floatX)


def genselect(arr, inds):
    """Select some elements along the inds.ndim-th axis of arr. Takes arr of
    shape shape (a1, a2, …, am), and inds of shape (a1, a2, …, a(k-1), bk),
    where k <= m. The elements of inds index into the kth axis of a, allowing
    you to keep only a subset of "supercolumns" along a given axis.

    There ought to be a Numpy function to do this, but I can't find it :/"""
    ishape = inds.shape
    irank = len(inds.shape)
    ashape = arr.shape
    arank = len(ashape)
    assert irank <= arank, 'indices dims cannot exceed array dims'
    assert ashape[:irank-1] == ishape[:-1], \
        'indices shape %s does not match array shape %s on all relevant axes' \
        % (ishape, ashape)
    oshape = ishape + ashape[-(arank - irank):]
    assert len(oshape) == arank
    out_volume = np.empty(oshape)
    # XXX: this is a really slow way of handling this. Is there a better way?
    mgrids = np.mgrid[[slice(None, top) for top in ishape[:-1]]]
    # flatten grids so that we can use them to index
    mgrids = [g.flatten() for g in mgrids]
    # could probably do this without the loop, but doesn't matter
    for i in range(ishape[-1]):
        ith_inds = inds[..., i].flatten()
        new_vals = arr[mgrids + [ith_inds]]
        exp_shape = (np.prod(ishape[:-1]), ) + ashape[-(arank - irank):]
        assert new_vals.shape == exp_shape, (new_vals.shape, exp_shape)
        out_volume[mgrids + [i]] = new_vals
    return out_volume


_t = None


def tprint(thing):
    # prints thing w/ displacement of time at beginning
    global _t
    now = time.time()
    if _t is None:
        print('[start t]' + str(thing))
    else:
        print('[+%.4fs]' % (now - _t) + str(thing))
    _t = now


def forecast_on_batch(dkf, poses, full_length, beam_size):
    """Extends a batch of pose sequences by the desired forecast length."""
    assert poses.ndim == 3, "Poses should be batch*time*dim"
    prefix_length = poses.shape[1]
    poses = fX(poses)
    batch_size = len(poses)
    # if this isn't true then we're not actually predicting anything (just
    # returning a truncated prefix of the original sequence passed through the
    # DKF)
    assert full_length > prefix_length
    # number of extra samples to produce for each beam item at each iteration
    extra_samples = 3

    tprint('startf')
    forecast = np.zeros(
        (batch_size, full_length, poses.shape[-1]), dtype=config.floatX)
    mask = np.ones_like(poses, dtype=config.floatX)
    init_zs = []
    init_xs = []
    init_err_mats = []
    _, mu_z, logcov_z = DKF_evaluate.infer(dkf, poses, mask)
    mu_z = fX(mu_z)
    for beam_step in range(beam_size * extra_samples):
        tprint('startf, step %d/%d' % (beam_step, beam_size * extra_samples))
        # ignore mu_z/logcov_z for now. will have to test later whether
        # using mu_z in place of z improves performance
        tent_init_z = fX(DKF_evaluate.sampleGaussian(mu_z, logcov_z))
        init_zs.append(tent_init_z)
        # store the result of applying the emission function to inferred zs
        tent_init_xs = fX(dkf.emission_fxn(tent_init_z))
        init_xs.append(tent_init_xs)
        err_vec = np.linalg.norm(tent_init_xs - poses, axis=-1).mean(axis=-1)
        init_err_mats.append(err_vec)
    # fill the beam
    tprint('bfill')
    best_starts = np.argsort(np.stack(init_err_mats, -1), axis=-1)
    best_starts = best_starts[:, :beam_size]
    assert best_starts.shape == (batch_size, beam_size)

    tprint('gs0')
    beam_zs = fX(genselect(np.stack(init_zs, axis=1), best_starts))
    exp_shape = (batch_size, beam_size, prefix_length)
    assert beam_zs.shape[:3] == exp_shape \
        and beam_zs.ndim == 4, (beam_zs.shape, exp_shape)

    tprint('gs1')
    # XXX: expensive! This is like 20s
    beam_xs = fX(genselect(np.stack(init_xs, axis=1), best_starts))
    assert beam_xs.shape[:3] == exp_shape \
        and beam_xs.ndim == 4, (beam_xs.shape, exp_shape)

    # now we can expand the beam one step at a time
    tprint('lstart')
    for t in range(prefix_length, full_length):
        tprint('iter %d' % t)
        new_x_lead_shape = (batch_size, beam_size * extra_samples, t + 1)
        # we only bother keeping one z
        new_z_lead_shape = (batch_size, beam_size * extra_samples, 1)
        # these will hold ALL the zs and xs for each of our beams
        beam_zs_cand = np.zeros(
            new_z_lead_shape + beam_zs.shape[-1:], dtype=config.floatX)
        beam_xs_cand = np.zeros(
            new_x_lead_shape + beam_xs.shape[-1:], dtype=config.floatX)
        for beam_ind in range(beam_size):
            tprint('iter %d, beam %d' % (t, beam_ind))
            this_beam_zs = beam_zs[:, beam_ind]
            this_beam_xs = beam_xs[:, beam_ind]
            this_beam_mu, this_beam_logcov \
                = dkf.transition_fxn(fX(this_beam_zs))
            for extra_ind in range(extra_samples):
                # XXX: for some reason the first loop through is super
                # expensive, but the later ones aren't really as expensive
                tprint('iter %d, beam %d, extra_ind %d' % (t, beam_ind,
                                                           extra_ind))
                this_ind_z = fX(
                    DKF_evaluate.sampleGaussian(this_beam_mu,
                                                this_beam_logcov))
                tprint('iter %d, beam %d, extra_ind %d, emis' % (t, beam_ind,
                                                                 extra_ind))
                this_ind_x = dkf.emission_fxn(this_ind_z)
                sub_ind = beam_ind * extra_samples + extra_ind
                tprint('iter %d, beam %d, extra_ind %d, asgn' % (t, beam_ind,
                                                                 extra_ind))
                beam_zs_cand[:, sub_ind, :] = this_ind_z
                beam_xs_cand[:, sub_ind, :-1] = this_beam_xs
                beam_xs_cand[:, sub_ind, -1:] = this_ind_x

        # now we can rank the candidates and throw out the ones we don't like
        # I'm going to use distance-from-last-prediction to figure this one out
        # Other things to consider ranking by:
        # - Norm of acceleration vector
        # - Distance from zero-velocity baseline
        # - …maybe some other stuff? IDK.

        # v acceleration
        # badness = np.linalg.norm(
        #     this_beam_xs[:, :, -1] - 2 * this_beam_xs[:, :, -2]
        #     + this_beam_xs[:, :, -3], axis=-1)
        # ^ acceleration

        # v velocity
        tprint('iter %d, badness' % t)
        badness = np.linalg.norm(
            beam_xs_cand[:, :, -1] - beam_xs_cand[:, :, -2], axis=-1)
        # ^ velocity

        by_badness = np.argsort(badness, axis=1)
        assert by_badness.shape == (batch_size, beam_size * extra_samples), \
            by_badness.shape
        beam_sel = by_badness[:, :beam_size]
        # new beam!
        # XXX: expensive! this takes like 20s; the one after is much faster
        tprint('iter %d, gs2' % t)
        beam_zs = genselect(beam_zs_cand, beam_sel)
        tprint('iter %d, gs3' % t)
        beam_xs = genselect(beam_xs_cand, beam_sel)

    # Can choose the beam element that looked best in the last step, or return
    # all. Will return all for now.
    # forecast = beam_xs[:, 0]

    tprint('lend')

    return forecast


def parse_dkf_args(runme_path, conf_path, weight_path):
    # DKF uses arguments like "-vm LR -infm structured", etc. This script
    # loads those arguments.
    new_argv = get_args(runme_path)
    new_argv.extend(['-reload', weight_path, '-params', conf_path])
    from parse_args_dkf import parse
    params = parse(new_argv)

    return params


def load_dkf(dataset, runme_path, conf_path, weight_path):
    # This is pretty much just copied from train.py. Mostly voodoo.
    params = parse_dkf_args(runme_path, conf_path, weight_path)

    # Add dataset and NADE parameters, which will become part of the model
    for k in ['dim_observations', 'data_type']:
        params[k] = dataset[k]
    if params['use_nade']:
        params['data_type'] = 'real_nade'

    # Remove from params
    removeIfExists('./NOSUCHFILE')
    reloadFile = params.pop('reloadFile')
    pfile = params.pop('paramFile')
    assert os.path.exists(pfile), pfile + ' not found. Need paramfile'
    dkf = DKF(params, paramFile=pfile, reloadFile=reloadFile)

    return dkf


def get_args(runme_path):
    """Parse a runme.sh script to get train.py arguments."""
    with open(runme_path, 'r') as fp:
        data = fp.read()
    tokens = shlex.split(data)
    # get everything after "train.py", omitting one-char whitespace (which
    # sometimes occur when you have "\<newline>")
    train_idx = tokens.index('train.py')
    assert train_idx >= 0
    rest = tokens[train_idx + 1:]
    no_nl = [t for t in rest if not (len(t) == 1 and t.isspace())]
    return no_nl


def get_all_preds(dkf, dataset, for_cond, for_pred, num_samples, is_2d,
                  pred_seq_ids, pred_frame_nums, cond_frame_nums):
    """Get a bunch of predictions for validation set. Tries to manage memory
    carefully!"""
    N, Tp, D = for_pred.shape
    Tc = for_cond.shape[1]
    T = Tp + Tc
    flat_samples = np.zeros((N, num_samples, T, D), dtype='float32')
    flat_samples = forecast_on_batch(dkf, for_cond, T, num_samples)
    # squash so that different samples appear in different rows
    by_row = flat_samples.reshape((N * num_samples, T, D))
    del flat_samples
    # gotta make this the same size as flat_samples
    all_frame_nums = np.concatenate([cond_frame_nums, pred_frame_nums], axis=1)
    assert all_frame_nums.shape[1] == T, all_frame_nums.shape
    all_frame_nums_flat = np.concatenate(
        [[r] * num_samples for r in all_frame_nums])
    pred_seq_ids_flat = np.concatenate(
        [[r] * num_samples for r in pred_seq_ids])
    if is_2d:
        rec_by_row = dataset.reconstruct_poses(by_row, pred_seq_ids_flat,
                                               all_frame_nums_flat)
    else:
        rec_by_row = dataset.reconstruct_skeletons(by_row, pred_seq_ids_flat)
    del by_row
    unflat_shape = (N, num_samples) + rec_by_row.shape[1:]
    unflat_rec = rec_by_row.reshape(unflat_shape)
    # return both predictions and conditioning prefix
    rv_cond = unflat_rec[:, :, :Tc]
    rv_pred = unflat_rec[:, :, Tc:]
    return rv_cond, rv_pred


parser = ArgumentParser()
parser.add_argument(
    '--num-samples',
    type=int,
    default=5,
    help='number of predictions to make for each test item')
parser.add_argument('runme_path', help='path to relevant runme.sh script')
parser.add_argument(
    'conf_path', help='path to *-config.pkl file in checkpoints')
parser.add_argument(
    'weight_path', help='path to *-params.h5 file in checkpoints')
parser.add_argument('dest_h5', help='.h5 file to write predictions to')

if __name__ == '__main__':
    args = parser.parse_args()

    print('Loading dataset')
    ds_dict = loadDataset()
    if 'p2d' in ds_dict:
        dataset = ds_dict['p2d']
    else:
        dataset = ds_dict['p3d']

    print('Loading DKF')
    dkf = load_dkf(ds_dict, args.runme_path, args.conf_path, args.weight_path)

    print('Generating eval data')
    is_2d = isinstance(dataset, p2d_loader.P2DDataset)
    pred_usable = None
    if is_2d:
        result = dataset.get_ds_for_eval(train=False, discard_no_annos=True)
        for_cond, for_pred = result['conditioning'], result['prediction']
        pred_scales = result['prediction_scales']
        cond_scales = result['conditioning_scales']
        if dataset.has_sparse_annos:
            pred_usable = result['prediction_valids']
        seq_ids = result['seq_ids']
        pred_frame_numbers = result['prediction_frame_nums']
        cond_frame_numbers = result['conditioning_frame_nums']
        for_pred_recon = dataset.reconstruct_poses(for_pred, seq_ids,
                                                   pred_frame_numbers)
        for_cond_recon = dataset.reconstruct_poses(for_cond, seq_ids,
                                                   cond_frame_numbers)
        pred_actions = result['prediction_actions']
        cond_actions = result['conditioning_actions']
        action_names = dataset.action_names
    else:
        for_cond, for_pred = dataset.get_ds_for_eval(train=False)
        for_pred_recon = dataset.reconstruct_skeletons(for_pred)
        pred_frame_numbers = cond_frame_numbers = None
    print('Getting predictions')
    dkf_cond, dkf_preds = get_all_preds(dkf, dataset, for_cond, for_pred,
                                        args.num_samples, is_2d, seq_ids,
                                        pred_frame_numbers, cond_frame_numbers)
    # # v XXX
    # all_inferred_z = np.concatenate(inferred_z_params, axis=0)
    # all_rolled_z = np.concatenate(rolled_z_params, axis=0)
    # np.savez('z_dist_params.npz', inferred=all_inferred_z,
    #          rolled=all_rolled_z)
    # # ^ XXX
    print('Writing predictions')
    with h5py.File(args.dest_h5, 'w') as fp:
        extra_data = {}
        fp['/method_name'] = 'dkf'
        if not is_2d:
            fp['/parents_3d'] = dataset.parents
            fp.create_dataset(
                '/skeletons_3d_true',
                compression='gzip',
                shuffle=True,
                data=for_pred_recon)
            fp.create_dataset(
                '/skeletons_3d_pred',
                compression='gzip',
                shuffle=True,
                data=dkf_preds)
            # fp.create_dataset(
            #     '/seq_ids_3d_json',
            #     compression='gzip',
            #     data=json.dumps(seq_ids.tolist()))
            # fp.create_dataset(
            #     '/orig_frame_numbers_3d',
            #     compression='gzip',
            #     data=orig_frame_numbers)
        else:
            fp['/parents_2d'] = dataset.parents
            fp.create_dataset(
                '/poses_2d_true',
                compression='gzip',
                shuffle=True,
                data=for_pred_recon)
            fp.create_dataset(
                '/poses_2d_pred',
                compression='gzip',
                shuffle=True,
                data=dkf_preds)
            fp['/scales_2d'] = f32(pred_scales)
            extra_data['pck_joints'] = dataset.pck_joints

            # I'm also going to add the *prefix* which we used to generate the
            # predictions, as well as the result of passing the conditioning
            # sequence through the DKF (will be sorta lossy)
            fp.create_dataset(
                '/poses_2d_cond_true',
                compression='gzip',
                shuffle=True,
                data=for_cond_recon)
            fp.create_dataset(
                '/poses_2d_cond_pred',
                compression='gzip',
                shuffle=True,
                data=dkf_cond)
            fp['/scales_2d_cond'] = f32(cond_scales)

            # next two params are for making videos of predictions on top of
            # original frames

            # tells us name of sequence in original file which contained poses
            # used for prediction
            fp['/seq_ids_2d_json'] = json.dumps(seq_ids.tolist())
            # tells us the sequence number of each frame in that original file
            fp.create_dataset(
                '/pred_frame_numbers_2d',
                compression='gzip',
                data=pred_frame_numbers)
            fp.create_dataset(
                '/cond_frame_numbers_2d',
                compression='gzip',
                data=cond_frame_numbers)

            # also action data
            fp.create_dataset(
                '/cond_actions_2d', compression='gzip', data=cond_actions)
            fp.create_dataset(
                '/pred_actions_2d', compression='gzip', data=pred_actions)
            fp['/action_names'] = json.dumps(action_names)
        if pred_usable is not None:
            fp['/is_usable'] = pred_usable
        fp['/extra_data'] = json.dumps(extra_data)
