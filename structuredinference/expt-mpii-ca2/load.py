"""Load MPII CA2 poses and associated actions."""

import numpy as np

import addpaths
from p2d_loader import P2DDataset


def loadDataset():
    seq_length = 32
    # skip of 1 because it's already downsapmled aggressively
    seq_skip = 1
    gap = 3
    data = P2DDataset(
        './mpii_ca2.h5',
        seq_length,
        seq_skip,
        gap=gap,
        val_frac=0.2,
        have_actions=True,
        completion_length=256,
        aclass_full_length=96,
        aclass_act_length=8,
        head_vel=False)

    # TODO: factor this out into common code (it's shared with IkeaDB and will
    # probably end up shared with Penn)
    dataset = {}
    dataset['dim_observations'] = data.dim_obs
    dataset['data_type'] = 'real'
    dataset['p2d'] = data

    dataset['train'], dataset['mask_train'], dataset['train_cond_vals'] \
        = data.get_pose_ds(train=True)
    dataset['val'], dataset['mask_val'], dataset['val_cond_vals'] \
        = dataset['test'], dataset['mask_test'], dataset['test_cond_vals'] \
        = data.get_pose_ds(train=False)
    # for action prediction
    dataset['train_aclass_ds'] = data.get_aclass_ds(train=True)
    dataset['val_aclass_ds'] = data.get_aclass_ds(train=False)
    # for sequence completion
    dataset['train_completions'] = data.get_completion_ds(train=True)
    dataset['val_completions'] = data.get_completion_ds(train=False)

    print('Shapes of various things:')
    to_check_shape = [
        'train', 'val', 'test', 'train_cond_vals', 'val_cond_vals',
        'test_cond_vals'
    ]
    for to_shape in to_check_shape:
        print('%s: %s' % (to_shape, dataset[to_shape].shape))
    to_check_len = [
        'train_aclass_ds', 'val_aclass_ds', 'train_completions',
        'val_completions'
    ]
    for name in to_check_len:
        print('%s: %d (list)' % (name, len(dataset[name])))

    return dataset

if __name__ == '__main__':
    dataset = loadDataset()
    print('Load successful')
