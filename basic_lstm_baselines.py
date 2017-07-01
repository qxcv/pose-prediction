#!/usr/bin/env python3
"""ERD and LSTM-3LR baselines for pose prediction."""

import sys
sys.path.append('keras')

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from keras.models import Model, load_model  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import CUSTOM_OBJECTS  # noqa: E402
from p2d_loader import P2DDataset, P3DDataset  # noqa: E402
import predict_erd_3d as perd  # noqa: E402
import predict_lstm_3lr as pl3lr  # noqa: E402
import predict_lstm as plstm  # noqa: E402
import predict_gan as pgan  # noqa: E402

MASK_VALUE = 0.0


def f32(x):
    return np.asarray(x, dtype='float32')


def mask_data(data, mask, mask_value=0.0):
    """Apply mask to data tensor by setting all masked time steps (second-last
    dimension) to zero."""
    assert data.ndim == 3, "Expected N*T*D data, got %s" % (data.shape, )
    # might have to change this if the mask is actually 2d
    # I forget what I actually chose
    assert mask.shape == data.shape, "Expected %s mask, got %s" % (data.shape,
                                                                   mask.shape)
    out_data = f32(data.copy())
    invalid_steps = ~mask.all(axis=-1, keepdims=True)
    invalid_steps_bc = np.broadcast_to(invalid_steps, out_data.shape)
    out_data[invalid_steps_bc] = mask_value
    return out_data


def train_model(module, model_path, dataset):
    # begin by making model which we will train
    seq_length = dataset.eval_condition_length + dataset.eval_test_length
    gap = dataset.eval_seq_gap
    discard_shorter = False

    # now get actual dataset for training
    if dataset.is_3d:
        train_data = mask_data(
            *dataset.get_ds_for_train(True, seq_length, gap, discard_shorter),
            mask_value=MASK_VALUE)
        val_data = mask_data(
            *dataset.get_ds_for_train(False, seq_length, gap, discard_shorter),
            mask_value=MASK_VALUE)
    else:
        train_data = mask_data(
            *dataset.get_ds_for_train(True, seq_length, gap, discard_shorter),
            mask_value=MASK_VALUE)
        val_data = mask_data(
            *dataset.get_ds_for_train(False, seq_length, gap, discard_shorter),
            mask_value=MASK_VALUE)

    # Y is one step ahead of X
    train_X, train_Y = train_data[:, :-1], train_data[:, 1:]
    val_X, val_Y = val_data[:, :-1], val_data[:, 1:]

    model = module.train_model(
        train_X,
        train_Y,
        val_X,
        val_Y,
        mask_value=MASK_VALUE,
        save_path=model_path)

    return model


def generic_caching_baseline(module, identifier, cache_dir, dataset):
    """Trains a model (if necessary), then predicts results on the supplied
    dataset."""
    model_path = os.path.join(cache_dir, 'model-%s.h5' % identifier)
    # convert the training model to an eval model
    if os.path.exists(model_path):
        print("Loading model for %s from '%s'" % (identifier, model_path))
        if hasattr(module, 'load_eval_model'):
            # GANs (and maybe other models) have weird loaders because they
            # save many models at train time (i.e. generator and discriminator)
            mod_train = module.load_eval_model(model_path, CUSTOM_OBJECTS)
        else:
            mod_train = load_model(model_path, CUSTOM_OBJECTS)
    else:
        print("Training model for %s anew (will save to '%s')" %
              (identifier, model_path))
        mod_train = train_model(module, model_path, dataset)

    mod_pred = module.make_model_predict(mod_train, mask_value=MASK_VALUE)
    if isinstance(mod_pred, Model):
        # Get a prediction wrapper, in the case that we just have a
        # step-by-step Keras model to do prediction. Pass it some conditioning
        # poses and a number of steps to predict, and it will return you that
        # many predicted steps (MAGIC!)
        def wrapper(in_tensor, steps_to_predict):
            assert in_tensor.ndim == 3, \
                "Expecting N*T*D tensor, got %s" % (in_tensor.shape,)
            # model batch size and number of steps is 1/1
            # TODO: if this is slow then you will have to fix it by increasing
            # batch size and increasing number of steps, then writing a wrapper
            # to put everything in the correct shape. Keras isn't very helpful
            # here :/
            out_shape = (in_tensor.shape[0], steps_to_predict,
                         in_tensor.shape[2])
            out_tensor = np.zeros(out_shape, dtype='float32')
            for n in tqdm(range(in_tensor.shape[0])):
                mod_pred.reset_states()
                # condition on input
                for t in range(in_tensor.shape[1]):
                    input = in_tensor[n:n + 1, t:t + 1]
                    result = mod_pred.predict_on_batch(input)
                # put only last prediction from input sequence in tensor
                out_tensor[n:n + 1, :1] = result
                # now predict rest of output sequence
                for t in range(1, steps_to_predict):
                    input = out_tensor[n:n + 1, t - 1:t]
                    result = mod_pred.predict_on_batch(input)
                    out_tensor[n:n + 1, t:t + 1] = result
            return out_tensor
        wrapper.method_name = identifier
        return wrapper
    # otherwise it's handled for us
    if not hasattr(mod_pred, 'method_name'):
        mod_pred.method_name = identifier
    return mod_pred


def write_baseline(cache_dir, dataset, steps_to_predict, method):
    meth_name = method.method_name
    out_path = os.path.join(cache_dir, 'results_' + meth_name + '.h5')
    print('Writing %s baseline to %s' % (meth_name, out_path))

    extra_data = {}
    if dataset.is_3d:
        cond_on, pred_on = dataset.get_ds_for_eval(train=False)
        pred_on_orig = f32(dataset.reconstruct_skeletons(pred_on))
        pred_usable = pred_scales = None
    else:
        evds = dataset.get_ds_for_eval(train=False)
        cond_on = evds['conditioning']
        pred_on = evds['prediction']
        pred_scales = evds['prediction_scales']
        cond_scales = evds['conditioning_scales']
        if dataset.has_sparse_annos:
            # hmm, the next key might be wrong (haven't tested)
            pred_usable = evds['prediction_valids']
        else:
            pred_usable = None
        seq_ids = evds['seq_ids']
        pred_frame_numbers = evds['prediction_frame_nums']
        cond_frame_numbers = evds['conditioning_frame_nums']
        extra_data['pck_joints'] = dataset.pck_joints
        pred_on_orig = f32(
            dataset.reconstruct_poses(pred_on, seq_ids, pred_frame_numbers))
        cond_on_orig = f32(
            dataset.reconstruct_poses(cond_on, seq_ids, cond_frame_numbers))

    if pred_usable is None:
        pred_usable = np.ones(pred_on_orig.shape[:2], dtype=bool)

    if pred_scales is None:
        pred_scales = np.ones(pred_on_orig.shape[:2], dtype='float32')

    result = method(cond_on, steps_to_predict)
    if dataset.is_3d:
        result = f32(dataset.reconstruct_skeletons(result))
    else:
        result = f32(
            dataset.reconstruct_poses(result, seq_ids, pred_frame_numbers))
    # insert an extra axis
    result = result[:, None]
    assert (result.shape[0],) + result.shape[2:] == pred_on_orig.shape, \
        (result.shape, pred_on_orig.shape)
    with h5py.File(out_path, 'w') as fp:
        fp['/method_name'] = meth_name
        if dataset.is_3d:
            fp['/parents_3d'] = dataset.parents
            fp.create_dataset(
                '/skeletons_3d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on_orig)
            fp.create_dataset(
                '/skeletons_3d_pred',
                compression='gzip',
                shuffle=True,
                data=result)
        else:
            fp['/parents_2d'] = dataset.parents
            fp.create_dataset(
                '/poses_2d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on_orig)
            fp.create_dataset(
                '/poses_2d_cond_true',
                compression='gzip',
                shuffle=True,
                data=cond_on_orig)
            fp['/scales_2d'] = f32(pred_scales)
            fp['/scales_2d_cond'] = f32(cond_scales)
            # TODO: is there a more meaningful notion of "predicted
            # conditioning pose" (poses_2d_cond_pred) for these models? For the
            # DKF we basically need to push the whole conditioning sequence
            # through a DKF, so we can see its repro of the prefix sequence.
            fp.create_dataset(
                '/poses_2d_pred',
                compression='gzip',
                shuffle=True,
                data=result)
            # make samples axis match for originals
            new_cond_orig_shape = (1, ) + result.shape[1:2] \
                + (1, ) * (cond_on_orig.ndim - 1)
            cond_on_tiled = np.tile(cond_on_orig[:, None], new_cond_orig_shape)
            fp.create_dataset(
                '/poses_2d_cond_pred',
                compression='gzip',
                shuffle=True,
                data=cond_on_tiled)
            fp.create_dataset(
                '/pred_frame_numbers_2d',
                compression='gzip',
                data=pred_frame_numbers)
            fp.create_dataset(
                '/cond_frame_numbers_2d',
                compression='gzip',
                data=cond_frame_numbers)
            fp['/seq_ids_2d_json'] = json.dumps(seq_ids.tolist())
        fp['/is_usable'] = pred_usable
        fp['/extra_data'] = json.dumps(extra_data)


NETWORK_MODULES = {
    'erd': perd,
    'lstm3lr': pl3lr,
    'lstm': plstm,
    'gan': pgan,
}
NETWORK_MODULE_CHOICES = sorted(NETWORK_MODULES.keys())

parser = argparse.ArgumentParser()
parser.add_argument(
    'model_type',
    choices=NETWORK_MODULE_CHOICES,
    help='type of model to train')
parser.add_argument('dataset_path', help='path to input HDF5 file')
parser.add_argument('output_dir', help='dir for output files and models')
parser.add_argument(
    '--3d',
    action='store_true',
    dest='is_3d',
    default=False,
    help='treat this as a 3D dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    cache_dir = args.output_dir
    try:
        os.makedirs(cache_dir)
    except FileExistsError:
        pass

    if args.is_3d:
        dataset = P3DDataset(args.dataset_path)
    else:
        dataset = P2DDataset(args.dataset_path, 32)

    # will be passed to other fns to train
    identifier = args.model_type
    module = NETWORK_MODULES[identifier]
    baseline = generic_caching_baseline(module, identifier, cache_dir, dataset)

    #  write_baseline(cache_dir, dataset, steps_to_predict, method):
    write_baseline(cache_dir, dataset, dataset.eval_test_length, baseline)
