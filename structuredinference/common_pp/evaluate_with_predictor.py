#!/usr/bin/env python3

"""Creates """

from act_pre_common import balance_aclass_ds, merge_actions, classifier_transform
dataset['val_aclass_ds'] = data.get_aclass_ds(train=False)
val_aclass_ds = dataset['val_aclass_ds']
# this isn't the only merge map! there's also a merge map which crushes everything down to *tach
# and spin (it probably keeps the flip action, incidentally)
merge_map = {'attach leg 1': 'attach leg', 'attach leg 2': 'attach leg', 'attach leg 3': 'attach leg',
             'attach leg 4': 'attach leg', 'detach leg 1': 'detach leg', 'detach leg 2': 'detach leg',
             'detach leg 3': 'detach leg', 'detach leg 4': 'detach leg', 'n/a': None}
aclass_target_names, val_aclass_ds = merge_actions(val_aclass_ds, merge_map, act_names)
val_aclass_ds_bal = balance_aclass_ds(val_aclass_ds, aclass_target_names)
true_X_clean = classifier_transform(true_X)
true_Y_pred_prob = action_model.predict(true_X_clean)
# data munging comes after this, pretty conceptually straightforward
