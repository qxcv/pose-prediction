from p2d_loader import P3DDataset


def loadDataset():
    fr = 30
    fr_eff = 15
    # sequences will be around 6s (90 frames); shorter ones get kept, but end
    # is masked out
    seq_length = 6 * fr_eff
    # drop anything shorter than a few hundred ms
    discard_shorter = int(round(0.8 * fr_eff))
    # jump forward this far between chosen sequences
    gap = 23
    data = P3DDataset(data_file_path='./ntu_data.h5')

    dataset = {}

    dataset['dim_observations'] = data.dim_obs
    dataset['data_type'] = 'real'
    dataset['p3d'] = data

    dataset['train'], dataset['mask_train'] \
        = data.get_ds_for_train(train=True, seq_length=seq_length,
                                discard_shorter=discard_shorter, gap=gap)
    dataset['val'], dataset['mask_val'] \
        = dataset['test'], dataset['mask_test'] \
        = data.get_ds_for_train(train=False, seq_length=seq_length,
                                discard_shorter=discard_shorter, gap=gap)

    print('Shapes of various things:')
    to_check_shape = ['train', 'val', 'test']
    for to_shape in to_check_shape:
        print('%s: %s' % (to_shape, dataset[to_shape].shape))

    return dataset
