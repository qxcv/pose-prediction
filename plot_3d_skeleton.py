#!/usr/bin/env python3
"""Plot a 3D skeleton from a HDF5 file (probably produced by
convert_ntu.py)."""

from argparse import ArgumentParser
from random import choice, randint

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from expmap import expmap_to_xyz, plot_xyz_skeleton

parser = ArgumentParser()
parser.add_argument('file_path', help='path to HDF5 file')

if __name__ == '__main__':
    args = parser.parse_args()
    with h5py.File(args.file_path, 'r') as fp:
        vid_names = list(fp['/seqs3d'].keys())
        chosen_vid = choice(vid_names)
        skeletons = fp['/seqs3d/' + chosen_vid + '/skeletons'].value
        parents = fp['/parents_3d'].value
        bone_lengths = fp['/bone_lengths_3d'].value

    # plot at 30/skip fps
    skip = 15
    plot_len = 4
    chosen_frame = randint(0, skeletons.shape[0] - 1 - plot_len * skip)
    print('Picked frames %d-%d of %s' % (chosen_frame, chosen_frame +
                                         (plot_len - 1) * skip, chosen_vid))
    fig = plt.figure()
    for i in range(0, plot_len):
        fn = chosen_frame + i * skip
        exp_skeleton = skeletons[fn]
        xyz_skeleton = expmap_to_xyz(exp_skeleton[None, ...], parents,
                                     bone_lengths)[0]
        ax = fig.add_subplot(1, plot_len, i + 1, projection='3d')
        plot_xyz_skeleton(xyz_skeleton, parents, ax)
        ax.set_title('Frame %d' % fn)
    plt.show()
