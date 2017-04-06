#!/usr/bin/env python3
"""Plot a 2D pose from a HDF5 file (probably produced by convert_ntu.py)."""

from argparse import ArgumentParser
from random import choice, randint

import h5py
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('file_path', help='path to HDF5 file')

if __name__ == '__main__':
    args = parser.parse_args()
    with h5py.File(args.file_path, 'r') as fp:
        vid_names = list(fp['/seqs'].keys())
        chosen_vid = choice(vid_names)
        poses = fp['/seqs/' + chosen_vid + '/poses'].value
        parents = fp['/parents'].value

    # plot at 30/skip fps
    skip = 15
    plot_len = 4
    num_poses = poses.shape[0]
    end = max(poses.shape[0] - plot_len * skip, 0)
    chosen_frame = randint(0, end)
    print('Picked frames %d-%d of %s' % (chosen_frame, chosen_frame +
                                         (plot_len - 1) * skip, chosen_vid))
    fig = plt.figure()
    for i in range(0, plot_len):
        fn = chosen_frame + i * skip
        pose = poses[fn]
        ax = fig.add_subplot(1, plot_len, i + 1)
        for child, parent in enumerate(parents):
            if child == parent:
                continue
            coords = pose[:, (child, parent)]
            ax.plot(coords[0], coords[1], 'b-')
        ax.set_aspect('equal')
        ax.set_title('Frame %d' % fn)
        ax.invert_yaxis()
    plt.show()
