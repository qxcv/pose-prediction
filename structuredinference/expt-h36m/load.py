import numpy as np

from h36m_loader import load_data


def loadDataset(use_cond=False):
    seq_length = 32
    seq_skip = 3
    if use_cond:
        train_X, val_X, mean, std, train_actions, val_actions, act_names \
            = load_data(seq_length, seq_skip, val_subj_5=False,
                        return_actions=True)
    else:
        train_X, val_X, mean, std \
            = load_data(seq_length, seq_skip, val_subj_5=False,
                        return_actions=False)
    dim_observations = train_X.shape[2]

    dataset = {}

    dataset['train'] = train_X
    dataset['mask_train'] = np.ones(train_X.shape[:2])

    dataset['valid'] = val_X
    dataset['mask_valid'] = np.ones(val_X.shape[:2])

    dataset['test'] = dataset['valid']
    dataset['mask_test'] = dataset['mask_valid']

    dataset['dim_observations'] = dim_observations
    dataset['data_type'] = 'real'

    dataset['h36m_mean'] = mean
    dataset['h36m_std'] = std

    if use_cond:
        dataset['train_cond_vals'] = train_actions
        dataset['val_cond_vals'] = val_actions
        dataset['test_cond_vals'] = val_actions
        dataset['h36m_action_names'] = act_names

    return dataset
