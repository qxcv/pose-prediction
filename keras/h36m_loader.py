"""Utilities for loading Human3.6M data. Should be free of
Keras/Theano/TensorFlow code, since it gets used from scripts that employ other
frameworks."""

import re
from glob import glob
import numpy as np
from multiprocessing import Pool
from os import path

# Indices of nonzero features in expmap mocap inds. I have no idea why the
# zeroed features are ever written out.
GOOD_MOCAP_INDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29,
    30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56,
    57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86
]
# Remainder of the 99 entries are constant zero (apparently)
TRUE_NUM_ENTRIES = 99
# Used to create one-hot representations of actions used in a sequence
ACTION_NAMES = [
    None, 'directions', 'discussion', 'eating', 'greeting', 'phoning',
    'posing', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto',
    'waiting', 'walking', 'walkingdog', 'walkingtogether'
]
ACTION_IDS = {name: index for index, name in enumerate(ACTION_NAMES)}
NUM_ACTIONS = len(ACTION_NAMES)


def insert_junk_entries(data):
    assert 3 >= data.ndim >= 2 and data.shape[-1] == len(GOOD_MOCAP_INDS)
    rv = np.zeros(data.shape[:-1] + (TRUE_NUM_ENTRIES, ))
    rv[..., GOOD_MOCAP_INDS] = data
    return rv


def prepare_file(filename, seq_length, seq_skip, gap):
    poses = np.loadtxt(filename, delimiter=',')
    assert poses.ndim == 2 and poses.shape[1] == 99, poses.shape

    zero_inds, = np.nonzero((poses != 0).any(axis=0))
    assert (zero_inds == GOOD_MOCAP_INDS).all(), zero_inds
    poses = poses[:, GOOD_MOCAP_INDS]

    seqs = []
    true_length = seq_length * seq_skip
    end = len(poses) - true_length + 1
    for start in range(0, end, gap):
        seqs.append(poses[start:start + true_length:seq_skip])
    X = np.stack(seqs)

    return X


def is_valid(data):
    return np.isfinite(data).all()


_fnre = re.compile(r'^expmap_S(?P<subject>\d+)_(?P<action>.+).txt.gz$')


def _mapper(arg):
    """Worker to load data in parallel"""
    filename, seq_length, seq_skip, gap = arg
    base = path.basename(filename)
    meta = _fnre.match(base).groupdict()
    subj_id = int(meta['subject'])
    X = prepare_file(filename, seq_length, seq_skip, gap)
    action_name = meta['action'].split('_')[0]
    action_id = ACTION_IDS[action_name]
    actions = np.full(X.shape[:2], action_id, dtype='uint8')

    return subj_id, filename, X, actions


def load_data(seq_length=32, seq_skip=3, val_subj_5=True,
              return_actions=False, gap=17):
    # seq_skip is downsample factor, seq_length is output length (after
    # downsampling), gap is number of source frames to step forward between
    # choosing outputs
    root = path.dirname(path.abspath(__file__))
    filenames = glob(path.join(root, 'h36m-3d-poses', 'expmap_*.txt.gz'))
    assert len(filenames) > 0, \
        "Need some pose sequences to read! Check h36m-3d-poses at %s" % root

    train_X_blocks = []
    test_X_blocks = []
    if return_actions:
        train_action_blocks = []
        test_action_blocks = []

    if not val_subj_5:
        # Need to make a pool of val_filenames
        all_inds = np.random.permutation(len(filenames))
        val_count = int(0.2 * len(all_inds) + 1)
        val_inds = all_inds[:val_count]
        fn_arr = np.array(filenames)
        val_filenames = set(fn_arr[val_inds])

    print('Spawning pool')
    pool = Pool()
    try:
        fn_seq = ((fn, seq_length, seq_skip, gap) for fn in filenames)
        for subj_id, filename, X, action_block in pool.map(_mapper, fn_seq):
            if val_subj_5:
                is_val = subj_id == 5
            else:
                is_val = filename in val_filenames
            if is_val:
                # subject 5 is for testing
                test_X_blocks.append(X)
                if return_actions:
                    test_action_blocks.append(action_block)
            else:
                train_X_blocks.append(X)
                if return_actions:
                    train_action_blocks.append(action_block)
    finally:
        pool.terminate()

    # Memory usage is right on the edge of what small machines are capable of
    # handling here, so I'm being careful to delete large unneeded structures.
    train_X = np.concatenate(train_X_blocks, axis=0)
    del train_X_blocks
    if return_actions:
        train_action_ids = np.concatenate(train_action_blocks, axis=0)
        del train_action_blocks
        train_actions = np.arange(NUM_ACTIONS) == train_action_ids[..., None]
        del train_action_ids
    test_X = np.concatenate(test_X_blocks, axis=0)
    del test_X_blocks
    if return_actions:
        test_action_ids = np.concatenate(test_action_blocks, axis=0)
        del test_action_blocks
        test_actions = np.arange(NUM_ACTIONS) == test_action_ids[..., None]
        del test_action_ids

    N, T, D = train_X.shape

    reshaped = train_X.reshape((N * T, D))
    mean = reshaped.mean(axis=0).reshape((1, 1, -1))
    std = reshaped.std(axis=0).reshape((1, 1, -1))
    del reshaped

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    assert is_valid(train_X)
    assert is_valid(test_X)

    ret_vals = train_X, test_X, mean, std
    if return_actions:  # make action block as well
        ret_vals += (train_actions, test_actions, ACTION_NAMES)

    return ret_vals
