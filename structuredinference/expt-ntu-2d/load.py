from p2d_loader import P2DDataset


def loadDataset():
    # frame rate is 30fps, but we skip frames so that we go down to 15fps
    fr_eff = 15
    # I'm going to keep a few seconds. Some sequences are shorter than others,
    # so there's a definite tradeoff here:
    seq_length = 2 * fr_eff
    # drop anything shorter than a few hundred ms
    discard_shorter = int(round(0.8 * fr_eff))
    # jump forward this far between chosen sequences
    gap = 23
    data = P2DDataset(
        data_file_path='./ntu_data.h5',
        seq_length=seq_length,
        gap=gap)

    dataset = {}

    dataset['dim_observations'] = data.dim_obs
    dataset['data_type'] = 'real'
    dataset['p2d'] = data

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
