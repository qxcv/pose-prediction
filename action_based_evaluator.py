"""Evaluates pose sequences based on how well an action classifier recognises
them. Oh, also trains the action classifier :)"""
import sys
sys.path.append('keras')

import argparse  # noqa: E402
import os  # noqa: E402
import json  # noqa: E402

from keras.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from keras.models import Sequential  # noqa: E402
from keras.layers import Bidirectional, GRU, Dropout, Dense, \
    GaussianNoise, TimeDistributed, Masking  # noqa: E402
from keras.regularizers import l2  # noqa: E402
import numpy as np  # noqa: E402

from p2d_loader import P2DDataset  # noqa: E402

MASK_VALUE = 0.0


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
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform',
                name='gru1'),
            name='bidi1'))
    model.add(Dropout(0.2, name='drop1'))

    model.add(
        GRU(50,
            return_sequences=True,
            dropout_W=0.2,
            dropout_U=0.2,
            W_regularizer=l2(0.001),
            activation='relu',
            init='uniform',
            name='gru2'))
    model.add(Dropout(0.2, name='drop2'))

    model.add(
        TimeDistributed(
            Dense(
                50, W_regularizer=l2(0.001), activation='relu',
                name='dense1')))
    model.add(Dropout(0.3, name='drop3'))

    model.add(
        TimeDistributed(Dense(50, W_regularizer=l2(0.001), name='dense2')))
    model.add(Dropout(0.3, name='drop4'))

    model.add(
        TimeDistributed(
            Dense(num_actions, activation='softmax', name='dense3')))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def train_act_class_model(dataset, merge_map, balance_func=None):
    db = dataset['p2d']
    old_act_names = db.action_names
    _, train_aclass_ds \
        = merge_actions(dataset['train_aclass_ds'], merge_map, old_act_names)
    aclass_target_names, val_aclass_ds \
        = merge_actions(dataset['val_aclass_ds'], merge_map, old_act_names)
    train_aclass_ds_bal = balance_aclass_ds(
        train_aclass_ds, aclass_target_names, target_func=balance_func)
    # it's okay if the validation set isn't fully balanced in this case, since
    # some actions don't appear in it at all.
    val_aclass_ds_bal = balance_aclass_ds(
        val_aclass_ds, aclass_target_names, target_func=balance_func)
    n_actions = len(aclass_target_names)
    print('Number of actions: %d' % n_actions)
    print('Actions: ' + ', '.join(aclass_target_names))
    train_X, train_Y = to_XY(train_aclass_ds_bal, n_actions)
    val_X, val_Y = to_XY(val_aclass_ds_bal, n_actions)

    print('Action balance (train): ', train_Y.sum(axis=0))
    print('Action balance (val): ', val_Y.sum(axis=0))

    checkpoint_dir = './chkpt-aclass/'
    try:
        os.makedirs(checkpoint_dir)
    except FileExistsError:
        pass

    meta_path = os.path.join(checkpoint_dir, 'meta.json')
    with open(meta_path, 'w') as fp:
        to_dump = {
            'actions': list(aclass_target_names),
        }
        json.dump(to_dump, fp)

    seq_len, num_channels = train_X.shape[1:]
    num_actions = val_Y.shape[1]
    model = make_model(seq_len, num_channels, num_actions)
    model.fit(
        train_X,
        train_Y,
        batch_size=64,
        nb_epoch=1000,
        validation_data=(val_X, val_Y),
        callbacks=[
            EarlyStopping(monitor='val_acc', patience=50), ModelCheckpoint(
                checkpoint_dir +
                'action-classifier-{epoch:02d}-{val_loss:.2f}.hdf5',
                save_best_only=True)
        ],
        shuffle=True)


def one_hot_cat(array, num_choices):
    # converts integer  class vector ot one-hot
    rng = np.arange(num_choices)[None, ...]
    rv = array[..., None] == rng
    assert rv.shape == array.shape + (num_choices, )
    assert (np.sum(rv, axis=-1) == 1).all()
    return rv


def one_hot_max(array):
    # suppresses all but the maximum entry in each row
    # (or last dimension for >2d tensors)
    match_inds = np.argmax(array, axis=-1)
    return one_hot_cat(match_inds, array.shape[-1])


def balance_aclass_ds(aclass_ds, act_names, target_func=None):
    # find appropriate number of samples for a single action class,
    # then trim "heavy" action classes to have no more than
    # that number of samples
    class_map = np.zeros((len(aclass_ds), len(act_names)))
    for ds_idx, item in enumerate(aclass_ds):
        class_num = item['action_label']
        class_map[ds_idx, class_num] = 1
    support = class_map.sum(axis=0)
    if target_func is None:
        target_func = np.min
    support_target = int(target_func(support))
    to_keep = np.zeros((len(aclass_ds), ))
    for class_num in range(len(act_names)):
        if support[class_num] <= support_target:
            to_keep[class_map[:, class_num] == 1] = 1
        else:
            # drop all but [:median_support] of these
            class_inds, = np.nonzero(class_map[:, class_num])
            perm = np.random.permutation(len(class_inds))[:support_target]
            chosen_inds = class_inds[perm]
            to_keep[chosen_inds] = 1
    rv = []
    for choose_ind in np.nonzero(to_keep)[0]:
        rv.append(aclass_ds[choose_ind])
    return rv


def merge_actions(aclass_ds, merge_map, act_names):
    for class_name in act_names:
        if class_name not in merge_map:
            merge_map[class_name] = class_name
    new_class_names = sorted({
        class_name
        for class_name in merge_map.values() if class_name is not None
    })
    new_class_nums = []
    for class_name in act_names:
        new_name = merge_map[class_name]
        if new_name is None:
            new_num = None
        else:
            new_num = new_class_names.index(new_name)
        new_class_nums.append(new_num)
    new_aclass_ds = []
    for item in aclass_ds:
        new_item = {}
        new_item.update(item)
        action = item['action_label']
        new_action = new_class_nums[action]
        if new_action is None:
            continue
        new_item['action_label'] = new_action
        new_aclass_ds.append(new_item)
        new_item['action_label_orig'] = action
    return new_class_names, new_aclass_ds


def classifier_transform(X):
    """Transforms pose block X for classification purposes. Really just
    augments existing representation with differences from previous time
    step."""
    T, D = X[0].shape
    X_delta = X[:, 1:] - X[:, :-1]
    X_cat = np.concatenate((X[:, 1:], X_delta), axis=2)
    assert X_cat.shape == (X.shape[0], X.shape[1] - 1,
                           X.shape[2] * 2), X_cat.shape
    return X_cat


def to_XY(ds, num_classes):
    """Convert action classification dataset to X, Y format that I can train
    classifier on."""
    # TODO: handle masking
    Y_ints = np.array([d['action_label'] for d in ds])
    Y = one_hot_cat(Y_ints, num_classes)

    # TODO: try to reconstruct poses before learning on them. Maybe subtract
    # out mean pose of each sequence.
    X = classifier_transform(np.stack((d['poses'] for d in ds), axis=0))

    return X, Y


# Basic Ikea merge map (use at least this!)
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
#
# More aggressive merge map for Ikea.
#
# merge_map = {
#     'attach leg 1': '*tach',
#     'attach leg 2': '*tach',
#     'attach leg 3': '*tach',
#     'attach leg 4': '*tach',
#     'detach leg 1': '*tach',
#     'detach leg 2': '*tach',
#     'detach leg 3': '*tach',
#     'detach leg 4': '*tach',
#     'spin in': 'spin',
#     'spin out': 'spin',
#     'n/a': None
# }

# For reference, here's the data processing sequence:
# 1) Reconstruct the poses in each sequence
# 2) Divide each pose by its (stored) scales
# 3) Centre the whole sequence around mean position of a single joint (maybe
#    the first one?)


def f32(arr):
    return arr.astype('float32')


def postprocess_poses(pose_sequence):
    assert pose_sequence.dim == 4 and pose_sequence.shape[2] == 2, \
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
        if not unmask_end:
            continue
        orig_pose_block[:unmask_end]
        frame_numbers = train_ds['frame_numbers'][seq_idx, :unmask_end]
        vid_name = train_ds['vid_names'][seq_idx]
        reconst_block = dataset.reconstruct_poses(
            orig_pose_block[None, ...],
            vid_names=[vid_name],
            frame_inds=[frame_numbers])
        # now postprocess and add padding
        postprocessed, = postprocess_poses(reconst_block)
        assert postprocessed.ndim == 2, \
            "should be 2D, got %s" % (postprocessed.shape,)
        pad_spec = [(0, orig_pose_block.shape[0] - postprocessed.shape[0]),
                    (0, 0)]
        padded = np.pad(
            postprocessed, pad_spec, 'constant', constant_value=MASK_VALUE)
        dest_pose_blocks.append(padded)

        # TODO: also make one hot vector of actions, with zeros for invalid
        # actions
    train_poses = np.stack(dest_pose_blocks, axis=0)

    # train_ds['poses']
    # train_ds['masks']
    # train_ds['actions']
    # train_ds['vid_names']
    # train_ds['frame_numbers']

    # evds = dataset.get_ds_for_eval(train=True)
    # cond_on = evds['conditioning']
    # pred_on = evds['prediction']
    # pred_scales = evds['prediction_scales']
    # cond_scales = evds['conditioning_scales']
    # pred_on_orig = f32(
    #     dataset.reconstruct_poses(pred_on, seq_ids, pred_frame_numbers))
    # cond_on_orig = f32(
    #     dataset.reconstruct_poses(cond_on, seq_ids, cond_frame_numbers))
    # pred_actions = evds['prediction_actions']
    # cond_actions = evds['conditioning_actions']
    # action_names = dataset.action_names


def eval(args):
    pass


parser = argparse.ArgumentParser(
    description='train action classifier and use it to evaluate poses')
parser.add_argument('work_dir', help='dir for output files and models')
subparsers = parser.add_subparsers()
parser_train = subparsers.add_parser('train')
parser.add_argument('dataset_path', help='path to input HDF5 file')
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
