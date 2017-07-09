#!/usr/bin/env python3
"""Evaluates pose sequences based on how well an action classifier recognises
them. Oh, also trains the action classifier :)"""
import sys
sys.path.append('keras')

import argparse  # noqa: E402
import os  # noqa: E402
import glob  # noqa: E402

from keras.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from keras.models import Sequential, load_model  # noqa: E402
from keras.layers import Bidirectional, GRU, Dropout, Dense, \
    GaussianNoise, TimeDistributed, Masking  # noqa: E402
from keras.regularizers import l2  # noqa: E402
import numpy as np  # noqa: E402
import h5py  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from p2d_loader import P2DDataset  # noqa: E402

MASK_VALUE = 0.0


def copy_weights(source, dest):
    assert len(source.layers) == len(dest.layers)
    for src_layer, dest_layer in zip(source.layers, dest.layers):
        assert dest_layer.__class__ == src_layer.__class__
        dest_layer.set_weights(src_layer.get_weights())


def make_model(seq_len, num_channels, num_actions):
    # model from Anoop
    model = Sequential()
    model.add(
        Masking(
            MASK_VALUE, input_shape=(seq_len, num_channels), name='masking'))
    model.add(GaussianNoise(0.05, name='gauss1'))
    model.add(
        Bidirectional(
            GRU(50,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001),
                activation='relu',
                kernel_initializer='uniform',
                name='gru1'),
            name='bidi1'))
    model.add(Dropout(0.2, name='drop1'))

    model.add(
        GRU(50,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2(0.001),
            activation='relu',
            kernel_initializer='uniform',
            name='gru2'))
    model.add(Dropout(0.2, name='drop2'))

    model.add(
        TimeDistributed(
            Dense(
                50,
                kernel_regularizer=l2(0.001),
                activation='relu',
                name='dense1')))
    model.add(Dropout(0.3, name='drop3'))

    model.add(
        TimeDistributed(
            Dense(50, kernel_regularizer=l2(0.001), name='dense2')))
    model.add(Dropout(0.3, name='drop4'))

    model.add(
        TimeDistributed(
            Dense(num_actions, activation='softmax', name='dense3')))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def one_hot_cat(array, num_choices):
    # converts integer  class vector ot one-hot
    rng = np.arange(num_choices)[None, ...]
    rv = array[..., None] == rng
    assert rv.shape == array.shape + (num_choices, )
    nonzero = np.sum(rv, axis=-1) == 1
    if not nonzero.all():
        print("WARNING: %d/%d one-hot vectors don't have an entry" %
              (np.sum(~nonzero), nonzero.size))
    return rv


def one_hot_argmax(array):
    # OLD VERSION:
    # suppresses all but the maximum entry in each row
    # (or last dimension for >2d tensors)
    # match_inds = np.argmax(array, axis=-1)
    # return one_hot_cat(match_inds, array.shape[-1])

    # NEW VERSION: just converts to indices
    return np.argmax(array, axis=-1)


# def balance_aclass_ds(aclass_ds, act_names, target_func=None):
#     # find appropriate number of samples for a single action class,
#     # then trim "heavy" action classes to have no more than
#     # that number of samples
#     class_map = np.zeros((len(aclass_ds), len(act_names)))
#     for ds_idx, item in enumerate(aclass_ds):
#         class_num = item['action_label']
#         class_map[ds_idx, class_num] = 1
#     support = class_map.sum(axis=0)
#     if target_func is None:
#         target_func = np.min
#     support_target = int(target_func(support))
#     to_keep = np.zeros((len(aclass_ds), ))
#     for class_num in range(len(act_names)):
#         if support[class_num] <= support_target:
#             to_keep[class_map[:, class_num] == 1] = 1
#         else:
#             # drop all but [:median_support] of these
#             class_inds, = np.nonzero(class_map[:, class_num])
#             perm = np.random.permutation(len(class_inds))[:support_target]
#             chosen_inds = class_inds[perm]
#             to_keep[chosen_inds] = 1
#     rv = []
#     for choose_ind in np.nonzero(to_keep)[0]:
#         rv.append(aclass_ds[choose_ind])
#     return rv

# def classifier_transform(X):
#     """Transforms pose block X for classification purposes. Really just
#     augments existing representation with differences from previous time
#     step."""
#     T, D = X[0].shape
#     X_delta = X[:, 1:] - X[:, :-1]
#     X_cat = np.concatenate((X[:, 1:], X_delta), axis=2)
#     assert X_cat.shape == (X.shape[0], X.shape[1] - 1,
#                            X.shape[2] * 2), X_cat.shape
#     return X_cat

# Basic Ikea merge map (use at least this, maybe a more aggressive one)
#
# merge_map = {
#     'attach leg 1': 'attach leg',
#     'attach leg 2': 'attach leg',
#     'attach leg 3': 'attach leg',
#     'attach leg 4': 'attach leg',
#     'detach leg 1': 'detach leg',
#     'detach leg 2': 'detach leg',
#     'detach leg 3': 'detach leg',
#     'detach leg 4': 'detach leg',
#     'n/a': None
# }

# For reference, here's the data processing sequence:
# 1) Reconstruct the poses in each sequence
# 2) Divide each pose by its (stored) scales
# 3) Centre the whole sequence around mean position of a single joint (maybe
#    the first one?)
merge_maps = {
    'ikea': {
        'attach leg 1': '*tach',
        'attach leg 2': '*tach',
        'attach leg 3': '*tach',
        'attach leg 4': '*tach',
        'detach leg 1': '*tach',
        'detach leg 2': '*tach',
        'detach leg 3': '*tach',
        'detach leg 4': '*tach',
        'spin in': 'spin',
        'spin out': 'spin',
        'n/a': None
    },
    'ntu': {
        # all these involve moving hand to head/face (rouhgly)
        'salute':
        'Touch head',
        'take off glasses':
        'Touch head',
        'take off a hat/cap':
        'Touch head',
        'brushing hair':
        'Touch head',
        'brushing teeth':
        'Touch head',
        'wear on glasses':
        'Touch head',
        'touch neck (neckache)':
        'Touch head',
        'touch head (headache)':
        'Touch head',
        'put on a hat/cap':
        'Touch head',
        'wipe face':
        'Touch head',
        'drink water':
        'Touch head',
        'make a phone call/answer phone':
        'Touch head',
        # whole body (including legs) moves up or down
        'standing up (from sitting position)':
        'Move body up/down',
        'falling':
        'Move body up/down',
        'jump up':
        'Move body up/down',
        'sitting down':
        'Move body up/down',
        'hopping (one foot jumping)':
        'Move body up/down',
        'pickup':
        'Move body up/down',
        # kinds of walking, basically
        'walking apart from each other':
        'Locomotion',
        'walking towards each other':
        'Locomotion',
        'staggering':
        'Locomotion',
        # interactions and other things which need hands to be outstretched
        "touch other person's pocket":
        'Hands out',
        'giving something to other person':
        'Hands out',
        'pushing other person':
        'Hands out',
        'punching/slapping other person':
        'Hands out',
        'point finger at the other person':
        'Hands out',
        'pointing to something with finger':
        'Hands out',
        'pat on back of other person':
        'Hands out',
        'hugging other person':
        'Hands out',
        'handshaking':
        'Hands out',
        # meant to be fine-grained actions with hands in front of body
        'check time (from watch)':
        'Other hand action',
        'put the palms together':
        'Other hand action',
        'clapping':
        'Other hand action',
        'hand waving':
        'Other hand action',
        'throw':
        'Other hand action',
        'rub two hands together':
        'Other hand action',
        'touch chest (stomachache/heart pain)':
        'Other hand action',
        'taking a selfie':
        'Other hand action',
        'touch back (backache)':
        'Other hand action',
        'tear up paper':
        'Other hand action',
        'use a fan (with hand or paper)/feeling warm':
        'Other hand action',
        'eat meal/snack':
        'Other hand action',
        'put something inside pocket / take out something from pocket':
        'Other hand action',
        'take off a shoe':
        'Other hand action',
        'playing with phone/tablet':
        'Other hand action',
        'typing on a keyboard':
        'Other hand action',
        'reading':
        'Other hand action',
        'writing':
        'Other hand action',
        'take off jacket':
        'Other hand action',
        'wear a shoe':
        'Other hand action',
        'wear jacket':
        'Other hand action',
        'cross hands in front (say stop)':
        'Other hand action',
        # two kicks, go figure
        'kicking other person':
        'Kicking',
        'kicking something':
        'Kicking',
        # not sure how to categorise this, so I'm throwing them out
        # (was going to make an "everything else" category, but that may not be
        # productive)
        'nod head/bow':
        'Everything else',
        'cheer up':
        'Everything else',
        'shake head':
        'Everything else',
        'sneeze/cough':
        'Everything else',
        'nausea or vomiting condition':
        'Everything else',
        'drop':
        'Everything else',
    },
}


def merge_actions(actions, act_names, dataset_name):
    assert actions.ndim == 2, \
        "actions should be N*T, but are %s" % (actions.shape,)

    merge_map = merge_maps.get(dataset_name)
    if merge_map is None:
        raise ValueError("don't know how to handle '%s' dataset" %
                         dataset_name)

    # create new_class_nums to tell us which old class IDs map to which new
    # ones
    for class_name in act_names:
        if class_name not in merge_map:
            merge_map[class_name] = class_name
    merged_act_names = sorted({
        class_name
        for class_name in merge_map.values() if class_name is not None
    })
    new_class_nums = []
    for class_name in act_names:
        new_name = merge_map[class_name]
        if new_name is None:
            new_num = None
        else:
            new_num = merged_act_names.index(new_name)
        new_class_nums.append(new_num)

    # sometimes we have out-of-range actions to indicate padding
    # figure out what those are
    uniq_vals = np.unique(actions)
    maybe_pads = uniq_vals[uniq_vals > 127]
    assert len(maybe_pads) <= 1, \
        'can have zero or one pads, got pads %s' % maybe_pads
    if maybe_pads:
        pad = maybe_pads[0]
    else:
        pad = None

    # now we can merge the original numeric action matrix
    placeholder = -245363367
    merged_actions = np.full_like(actions, placeholder)
    for orig_act_i in range(len(act_names)):
        new_num = new_class_nums[orig_act_i]
        if new_num is None:
            # get skipped yo
            new_num = pad if pad is not None else -1
        merged_actions[actions == orig_act_i] = new_num
    if pad is not None:
        merged_actions[merged_actions == pad] = pad
    assert not np.any(merged_actions.flatten() == placeholder), \
        "some actions were not replaced"

    return merged_actions, merged_act_names


def f32(arr):
    return arr.astype('float32')


def postprocess_poses(pose_sequence):
    assert pose_sequence.ndim == 4 and pose_sequence.shape[2] == 2, \
        "expected N*T*(XY)*J array"
    # average over joints, average over times
    mean_point = pose_sequence.mean(axis=-1).mean(axis=1)
    assert mean_point.shape == (pose_sequence.shape[0], 2)
    mean_shifted = pose_sequence - mean_point[:, None, :, None]
    # this should roughly approximate pose size across sequence
    scales = mean_shifted.reshape(mean_shifted.shape[:2] + (-1,)) \
        .std(axis=-1).mean(axis=1)
    rescaled = mean_shifted / scales[:, None, None, None]
    reshaped = rescaled.reshape(rescaled.shape[:2] + (-1, ))
    return reshaped


def train(args):
    dataset = P2DDataset(args.dataset_path, 32)
    train_length = dataset.eval_condition_length + dataset.eval_test_length
    if args.dataset_name == 'ntu':
        train_gap = max(dataset.eval_seq_gap, train_length)
    else:
        train_gap = max(dataset.eval_seq_gap, train_length // 2)
    train_ds = dataset.get_ds_for_train_extra(
        train=True,
        seq_length=train_length,
        gap=train_gap,
        discard_shorter=False)

    # now we've got to reconstruct (painful thanks to masks)
    orig_poses = train_ds['poses']
    dest_pose_blocks = []
    for seq_idx in range(len(train_ds['poses'])):
        orig_pose_block = orig_poses[seq_idx]

        # we will truncate the current sequence so that the crappy stuff is
        # masked
        mask = train_ds['masks'][seq_idx]
        amask = mask.reshape((len(mask), -1)).all(axis=-1)
        for unmask_end in range(len(mask)):
            if not amask[unmask_end]:
                break
        else:
            unmask_end = len(mask)
        if not unmask_end:
            # sometimes all steps are masked out for god knows what reason
            # ergo, just throw some empty crap into the queue
            padded = np.full((len(orig_pose_block), 2 * len(dataset.parents)),
                             MASK_VALUE)
        else:
            frame_numbers = train_ds['frame_numbers'][seq_idx, :unmask_end]
            vid_name = train_ds['vid_names'][seq_idx]
            reconst_block = dataset.reconstruct_poses(
                orig_pose_block[None, :unmask_end],
                vid_names=[vid_name],
                frame_inds=frame_numbers[None])
            # now postprocess and add padding
            postprocessed, = postprocess_poses(reconst_block)
            assert postprocessed.ndim == 2, \
                "should be 2D, got %s" % (postprocessed.shape,)
            pad_spec = [(0, orig_pose_block.shape[0] - postprocessed.shape[0]),
                        (0, 0)]
            padded = np.pad(
                postprocessed,
                pad_spec,
                'constant',
                constant_values=MASK_VALUE)
        dest_pose_blocks.append(padded)

    train_poses = np.stack(dest_pose_blocks, axis=0)

    # oh, and we can deal with actions
    merged_actions = train_ds['actions']
    merged_actions, merged_act_names = merge_actions(
        train_ds['actions'], dataset.action_names, args.dataset_name)
    oh_merged_actions = one_hot_cat(merged_actions, len(merged_act_names))

    assert oh_merged_actions.shape[:2] == train_poses.shape[:2]

    # _, train_aclass_ds \
    #     = merge_actions(dataset['train_aclass_ds'], merge_map, old_act_names)
    # aclass_target_names, val_aclass_ds \
    #     = merge_actions(dataset['val_aclass_ds'], merge_map, old_act_names)
    # train_aclass_ds_bal = balance_aclass_ds(
    #     train_aclass_ds, aclass_target_names, target_func=balance_func)
    # # it's okay if the validation set isn't fully balanced in this case,
    # # since some actions don't appear in it at all.
    # val_aclass_ds_bal = balance_aclass_ds(
    #     val_aclass_ds, aclass_target_names, target_func=balance_func)

    n_actions = len(merged_act_names)
    print('Number of actions: %d' % n_actions)
    print('Actions: ' + ', '.join(merged_act_names))

    # train_X, train_Y = to_XY(train_aclass_ds_bal, n_actions)
    # val_X, val_Y = to_XY(val_aclass_ds_bal, n_actions)
    # print('Action balance (train): ', train_Y.sum(axis=0))
    # print('Action balance (val): ', val_Y.sum(axis=0))

    checkpoint_dir = os.path.join(args.work_dir, 'chkpt-aclass')
    try:
        os.makedirs(checkpoint_dir)
    except FileExistsError:
        pass

    # meta_path = os.path.join(checkpoint_dir, 'meta.json')
    # with open(meta_path, 'w') as fp:
    #     to_dump = {
    #         'actions': list(merged_act_names),
    #     }
    #     json.dump(to_dump, fp)

    # gotta shuffle so that validation split is random
    data_perm = np.random.permutation(len(train_poses))
    train_poses = train_poses[data_perm]
    oh_merged_actions = oh_merged_actions[data_perm]

    seq_len, num_channels = train_poses.shape[1:]
    model = make_model(seq_len, num_channels, n_actions)
    model.fit(
        train_poses,
        oh_merged_actions,
        batch_size=256,
        nb_epoch=1000,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(monitor='val_acc', patience=50), ModelCheckpoint(
                os.path.join(
                    checkpoint_dir,
                    'action-classifier-{epoch:02d}-{val_loss:.2f}.hdf5'),
                save_best_only=True)
        ],
        shuffle=True)


def get_best_model(checkpoint_dir):
    model_names = glob.glob(
        os.path.join(checkpoint_dir, 'action-classifier-*.hdf5'))
    loss_name_pairs = []
    for model_name in model_names:
        acc_h5_part = model_name.rsplit('-', 1)[1]
        loss = float(acc_h5_part[:-len('.hdf5')])
        loss_name_pairs.append((loss, model_name))
    best_loss, best_name = min(loss_name_pairs)
    print("Loading model '%s' with loss %f" % (best_name, best_loss))
    return load_model(best_name)


def eval(args):
    print('Looking for model')
    checkpoint_dir = os.path.join(args.work_dir, 'chkpt-aclass')
    orig_model = get_best_model(checkpoint_dir)
    # input_shape is batch*time*channels
    _, _, num_chans = orig_model.input_shape
    _, _, num_acts = orig_model.output_shape

    print("Fetching results from '%s'" % args.results_file)
    with h5py.File(args.results_file, 'r') as fp:
        pred_poses = fp['/poses_2d_pred'].value
        true_poses = fp['/poses_2d_true'].value
        # need to reprocess the actions :/
        raw_actions = fp['/pred_actions_2d'].value

    assert pred_poses.ndim == 5, "expected 5D, got %s" % (pred_poses.shape)
    # we need to flatten out separate samples for pred_poses, then double
    # up true_poses, raw_actions, etc. as appropriate
    sample_dim = pred_poses.shape[1]
    new_pp_shape = (pred_poses.shape[0] * pred_poses.shape[1],
                    ) + pred_poses.shape[2:]
    pred_poses = pred_poses.reshape(new_pp_shape)
    # wooohoo broadcast hacks
    dupe_marker \
        = (np.arange(len(raw_actions)).reshape((-1, 1))
            + np.zeros((1, sample_dim), dtype='int')).flatten()
    assert dupe_marker.size == len(pred_poses), \
        "dupe_marker.shape %s, pred_poses.shape %s" \
        % (dupe_marker.shape, pred_poses.shape)
    raw_actions = raw_actions[dupe_marker]
    true_poses = true_poses[dupe_marker]
    new_seq_len = pred_poses.shape[1]
    assert pred_poses.shape == true_poses.shape, \
        "pred_poses.shape %s, true_poses.shape %s" \
        % (pred_poses.shape, true_poses.shape)

    print('Creating new model')
    new_model = make_model(new_seq_len, num_chans, num_acts)
    print('Initialising weights')
    copy_weights(orig_model, new_model)

    print('Loading dataset')
    dataset = P2DDataset(args.dataset_path, 32)
    merged_actions, merged_act_names = merge_actions(
        raw_actions, dataset.action_names, args.dataset_name)
    valid_acts = (0 <= merged_actions) \
        & (merged_actions < len(merged_act_names))
    print('%d/%d actions invalid (will be ignored)' % (
        (~valid_acts).flatten().sum(), valid_acts.size))

    def make_predictions(poses):
        postproc = postprocess_poses(poses)
        one_hot = new_model.predict(postproc)
        labels = one_hot_argmax(one_hot)

        # now do MNLL, accuracy
        flat_valid_flags = valid_acts.flatten()
        flat_labels = labels.flatten()[flat_valid_flags]
        flat_true_labels = merged_actions.flatten()[flat_valid_flags]
        flat_dists = one_hot.reshape((-1, one_hot.shape[-1]))[flat_valid_flags]

        accuracy = accuracy_score(flat_true_labels, flat_labels)
        f1 = f1_score(flat_true_labels, flat_labels, average='weighted')
        nll = -np.log(flat_dists[np.arange(len(flat_dists)), flat_labels])
        report = '\n'.join([
            'Accuracy: %f' % accuracy, 'F1: %f' % f1, 'MNLL: %f' % nll.mean()
        ])

        return report

    # TODO: maybe a better metric would show how much *worse* the predicted
    # actions are? Could measure "extra bits required to encode" or something
    # like that.

    print('How good do the predictions look?')
    pred_report = make_predictions(pred_poses)
    print(pred_report)

    print('How good do the originals look?')
    orig_report = make_predictions(true_poses)
    print(orig_report)

    dest_path = os.path.join(
        args.work_dir, os.path.basename(args.results_file) + '-results.txt')
    print("Writing results to '%s'" % dest_path)
    with open(dest_path, 'w') as fp:
        print("# Results for %s" % args.results_file, file=fp)
        print('\n## Predictions', file=fp)
        print(pred_report, file=fp)
        print('\n## Originals', file=fp)
        print(orig_report, file=fp)


parser = argparse.ArgumentParser(
    description='train action classifier and use it to evaluate poses')
parser.add_argument('work_dir', help='dir for output files and models')
parser.add_argument('dataset_path', help='path to input HDF5 file')
parser.add_argument(
    'dataset_name',
    choices=['ntu', 'ikea'],
    help='controls choice of action merge map to use ("ntu" or "ikea")')

subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser('train')
parser_train.set_defaults(func=train)

parser_eval = subparsers.add_parser('eval')
# model will be automatically read out of scratch directory, so no need to pass
# it explicitly
parser_eval.add_argument('results_file')
parser_eval.set_defaults(func=train)
parser_eval.set_defaults(func=eval)


def main():
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
