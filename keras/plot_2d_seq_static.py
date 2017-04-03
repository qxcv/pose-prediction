#!/usr/bin/env python2
"""Plots a sequence of 2D poses as a 3D plot."""

# Reminder: the only way to run Mayavi on your machine is to activate the
# "mayavi" virtualenv and run `ETS_TOOLKIT=wx ipython --gui=wx`. Note that the
# mayavi virtualenv is a Python 2 env, as Ubuntu 16.10's vtk package only
# supports Python 2.

import argparse

from mayavi import mlab
import numpy as np
from scipy.misc import imread


def mayavi_imshow(im):
    # See https://geoexamples.blogspot.com.au/2014_02_01_archive.html
    pass


def draw_poses(title,
               parents,
               pose_sequence,
               frame_paths=None,
               fps=50 / 3.0,
               crossover=None):
    T, _, J = pose_sequence.shape
    fig = mlab.figure()

    xmin, ymin = pose_sequence.min(axis=0).min(axis=1)
    xmax, ymax = pose_sequence.max(axis=0).max(axis=1)
    # ax.set_xlim(xmin, xmax)
    # ax.set_zlim(ymin, ymax)
    # ax.set_ylim(0, T/fps)
    # # ax.set_aspect('equal')
    # ax.invert_zaxis()

    if frame_paths:
        # Decide on plot indices ahead of time. Main constraints are:
        # - I don't want more than five frames in a 1s time period.
        # - I'd like to display the first and last frames
        inner_seconds = int(np.floor((T - 2) / fps))
        num_keyframes = min(T - 2, inner_seconds * 5) + 2
        skip = int(np.round(float(T) / num_keyframes))
        keyframe_inds = np.arange(0, T, skip)
        if keyframe_inds[-1] != T - 1:
            keyframe_inds = np.concatenate([keyframe_inds, [T - 1]])
        keyframe_inds = set(keyframe_inds)

    print('Keyframe inds: ', keyframe_inds)

    for t in range(T):
        # plot joints
        pose_xy_j = pose_sequence[t]
        depth = 10 * float(t) / fps
        for joint in range(1, len(parents)):
            joint_coord = pose_xy_j[:, joint]
            parent = parents[joint]
            parent_coord = pose_xy_j[:, parent]
            if (parent_coord == joint_coord).all():
                # mayavi will complain if we plot this zero-length line
                continue
            x_data = np.array((parent_coord[0], joint_coord[0]), dtype='int16')
            z_data = np.array((parent_coord[1], joint_coord[1]), dtype='int16')
            depth_data = np.full_like(x_data, depth, dtype='int16')
            x_data = np.array((parent_coord[0], joint_coord[0]))
            z_data = np.array((parent_coord[1], joint_coord[1]))
            depth_data = np.full_like(x_data, depth)
            mlab.plot3d(x_data, depth_data, z_data)

        # plot frame (TODO: only plot frames occasionally; don't want to overdo
        # it and make the plot unreadable)
        if frame_paths and t in keyframe_inds:
            im = imread(frame_paths[t])
            # mayavi wants a 2D array of integers instead of a real image; it
            # looks up colours in a colourmap
            im2d = np.arange(np.prod(im.shape[:2])).reshape(im.shape[:2])
            cmap = im.reshape((im.shape[0] * im.shape[1], im.shape[2]))
            # add alpha channel
            cmap = np.concatenate([
                cmap,
                np.full((cmap.shape[0], 1), 255, dtype='uint8')
            ], axis=1)
            assert cmap.dtype == np.uint8
            # extent = [xmin, xmax, ymin, ymax, zmin, zmaax]
            extent = np.array([0, im.shape[1], depth, depth, 0, im.shape[0]])
            ims = mlab.imshow(im2d, colormap='binary', extent=extent, opacity=0.5)
            # Documentation? What's that?
            # http://stackoverflow.com/a/24471211
            ims.module_manager.scalar_lut_manager.lut.table = cmap
            mlab.draw()

        # TODO:
        # - Back-connection to previous frames
        # - Thicken original pose tubes relative to back-connections
        # - Add crossover colour for predictions
        # - Make surex/y/z are going in the right direction, and that camera is
        #   in the right place
        # - Maybe plot action labels

    return fig


parser = argparse.ArgumentParser()
parser.add_argument('pose_path', help='path to .npz for poses')

if __name__ == '__main__':
    args = parser.parse_args()
    npz_path = args.pose_path
    print('Loading %s' % npz_path)
    loaded = np.load(npz_path)
    parents = loaded['parents']
    pose_keys = sorted([k for k in loaded.keys() if k.startswith('poses_')])
    print('Keys:', ', '.join(pose_keys))

    figures = []
    key = 'poses_train'
    figure = draw_poses(key, parents, loaded[key][0])
    figures.append((key, figure))
    # for key in pose_keys:
    #     figure = draw_poses(key, parents, loaded[key][0])
    #     figures.append((key, figure))

    mlab.show()
