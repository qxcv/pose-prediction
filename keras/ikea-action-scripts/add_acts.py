#!/usr/bin/env python3
"""Take a merged file and add action data. Create a new file along the way,
since I don't want to carry around silly 17GB action file."""

from argparse import ArgumentParser
from h5py import File, Group
import numpy as np

parser = ArgumentParser(description=__doc__)
parser.add_argument('src', help='path to source HDF5 file')
parser.add_argument('dest', help='path to dest HDF5 file')


def processor(dest):
    # We want to create a flat structure to make this file easy for Matlab to
    # consume. For example, actions associated with
    # '/2016-02-11/GOPR0030/frames' should go in a dataset named '/GOPR0030'
    # (in the destination). visited helps us check for dupe group names.
    visited = set()

    def process(group_name, group):
        if not isinstance(group, Group) or 'frames' not in group:
            # skip datasets
            return

        dirname = group_name.split('/')[-1]
        assert dirname not in visited, \
            "group names must be unique, but '%s' is not" % dirname
        visited.add(dirname)

        frames = group['frames']
        num_frames = frames.shape[0]
        print('Processing %d frames in %s' % (num_frames, group_name))
        act_in = frames[:, -12:]
        assert act_in.shape == (num_frames, 12)
        exps = np.exp(act_in)
        probs = exps / np.sum(exps, axis=1, keepdims=True).astype('float32')

        ds_name = '/' + dirname
        dest.create_dataset(ds_name, data=probs, chunks=True)

    return process


if __name__ == '__main__':
    args = parser.parse_args()
    with File(args.src, 'r') as src, File(args.dest, 'w-') as dest:
        process = processor(dest)
        src.visititems(process)
