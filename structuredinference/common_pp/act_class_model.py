import os
import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Bidirectional, GRU, Dropout, Dense, GaussianNoise
from keras.regularizers import l2

from common_pp.act_pre_common import merge_actions, to_XY, balance_aclass_ds


def make_model(seq_len, num_channels, num_actions):
    # model from Anoop
    model = Sequential()
    model.add(GaussianNoise(0.05, input_shape=(seq_len, num_channels), name='gauss1'))
    model.add(
        Bidirectional(
            GRU(50,
                return_sequences=True,
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform', name='gru1'), name='bidi1'))
    model.add(Dropout(0.2, name='drop1'))

    model.add(
            GRU(50,
                return_sequences=False,
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform', name='gru2'))
    model.add(Dropout(0.2, name='drop2'))

    model.add(Dense(50, W_regularizer=l2(0.001), activation='relu', name='dense1'))
    model.add(Dropout(0.3, name='drop3'))

    model.add(Dense(50, W_regularizer=l2(0.001), name='dense2'))
    model.add(Dropout(0.3, name='drop4'))

    model.add(Dense(num_actions, activation='softmax', name='dense3'))
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
    train_aclass_ds_bal = balance_aclass_ds(train_aclass_ds,
                                            aclass_target_names,
                                            target_func=balance_func)
    # it's okay if the validation set isn't fully balanced in this case, since
    # some actions don't appear in it at all.
    val_aclass_ds_bal = balance_aclass_ds(val_aclass_ds,
                                          aclass_target_names,
                                          target_func=balance_func)
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
