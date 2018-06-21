"""Common code for pose-from-action"""

import numpy as np


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
    """Transforms pose block X for classification purposes"""
    T, D = X[0].shape

    # add differences from previous time step
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
