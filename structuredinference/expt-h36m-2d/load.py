"""Load a 2D pose dataset (probably IkeaDB) and (optionally) associated
actions."""

from p2d_loader import P2DDataset


def loadDataset():
    seq_length = 32
    seq_skip = 3
    gap = 4
    data = P2DDataset(
        './h36m_action_data.h5',
        seq_length,
        have_actions=True,
        remove_head=False,
        head_vel=False)

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
