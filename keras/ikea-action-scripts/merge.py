#!/usr/bin/env python3
"""Merge together output vectors from activity recognition network."""

from argparse import ArgumentParser
from h5py import File, Group, Dataset
import numpy as np

parser = ArgumentParser(description=__doc__)
parser.add_argument(
    '--compress',
    default=False,
    action='store_true',
    help='use gzip compression')
parser.add_argument('src', help='path to source HDF5 file')
parser.add_argument('dest', help='path to dest HDF5 file')


def processor(dest, compress=False):
    if compress:
        ds_kwargs = {'compression': 9, 'shuffle': True}
    else:
        ds_kwargs = {}

    def process(group_name, group):
        if not isinstance(group, Group):
            # skip datasets
            return

        # Start by figuring out how many frames are in the current group
        num_frames = 0
        for child_name in group:
            child = group[child_name]
            if child_name.startswith('frame_') and isinstance(child, Dataset):
                frame_num = int(child_name.split('_')[1])
                assert frame_num >= 1
                num_frames = max(num_frames, frame_num)

        if num_frames <= 0:
            # skip groups with no 'frame_*' children
            return

        print('Processing %d frames in %s' % (num_frames, group_name))
        feat_width = len(group['frame_1'])
        feat_storage = np.full((num_frames, feat_width), np.nan, dtype='float32')
        for frame_num in range(1, num_frames + 1):
            to_store = group['frame_%d' % frame_num].value
            feat_storage[frame_num-1] = to_store

        assert not np.any(np.isnan(feat_storage.flatten()))

        ds_name = group_name + '/frames'
        dest.create_dataset(
            ds_name, data=feat_storage, chunks=True, **ds_kwargs)

    return process


if __name__ == '__main__':
    args = parser.parse_args()
    with File(args.src, 'r') as src, File(args.dest, 'w-') as dest:
        process = processor(dest, args.compress)
        src.visititems(process)
