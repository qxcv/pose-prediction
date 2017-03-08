#!/usr/bin/env python3
"""Tool for plotting sequences of generated 2D poses side-by-side."""

import argparse
import matplotlib.pyplot as plt
from matplotlib import animation as anim
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.misc import imread


def draw_poses(title, parents, pose_sequence, frame_paths=None,
               subplot_titles=None, fps=50/3.0, crossover=None):
    N, T, _, J = pose_sequence.shape
    fig = plt.figure()
    fig.suptitle(title)

    if frame_paths is not None:
        frames = {}
        for subseq_paths in frame_paths:
            for frame_path in subseq_paths:
                if frame_path not in frames:
                    frames[frame_path] = imread(frame_path)

    # Arrange plots in a square
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / float(cols)))
    subplots = []
    all_lines = []
    if frame_paths is not None:
        image_handles = []
    for spi in range(N):
        ax = fig.add_subplot(rows, cols, spi + 1)
        subplots.append(ax)

        sp_lines = []
        sp_joints = pose_sequence[spi]
        if frame_paths is None:
            xmin, ymin = sp_joints.min(axis=0).min(axis=1)
            xmax, ymax = sp_joints.max(axis=0).max(axis=1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.invert_yaxis()
        else:
            im = frames[frame_paths[spi][0]]
            image_handles.append(ax.imshow(im))

        for joint in range(1, len(parents)):
            joint_coord = sp_joints[0, :, joint]
            parent = parents[joint]
            parent_coord = sp_joints[0, :, parent]
            x_data = (parent_coord[0], joint_coord[0])
            y_data = (parent_coord[1], joint_coord[1])
            line, = ax.plot(x_data, y_data)
            sp_lines.append(line)

        if subplot_titles is not None:
            ax.set_title(subplot_titles[spi])

        all_lines.append(sp_lines)

    def animate_step(frame, joints, parents, all_lines):
        redrawn = []

        for spi in range(len(all_lines)):
            sp_joints = joints[spi]
            lines = all_lines[spi]
            # one line per edge, so no line for the parent
            assert len(parents) == len(lines) + 1

            if frame_paths is not None:
                im = frames[frame_paths[spi][frame]]
                handle = image_handles[spi]
                handle.set_data(im)
                redrawn.append(handle)

            for joint in range(1, len(parents)):
                joint_coord = sp_joints[frame, :, joint]
                parent = parents[joint]
                parent_coord = sp_joints[frame, :, parent]
                line = lines[joint - 1]
                line.set_xdata((parent_coord[0], joint_coord[0]))
                line.set_ydata((parent_coord[1], joint_coord[1]))
                redrawn.append(line)

            if crossover is not None:
                if frame >= crossover:
                    colour = 'blue'
                else:
                    colour = 'green'
                ax = subplots[spi]
                spines = list(ax.spines.values())
                for spine in spines:
                    spine.set_linewidth(6)
                    spine.set_color(colour)
                    redrawn.append(spine)

        return redrawn

    return anim.FuncAnimation(
        fig,
        animate_step,
        frames=T,
        repeat=True,
        fargs=(pose_sequence, parents, all_lines),
        interval=1000 * (1/fps),
        blit=True)


def video_worker(ks):
    key, seq = ks
    print('Working on', key)
    seq.save(key + '.mp4', writer='avconv', fps=50 / 3.0)
    return key


parser = argparse.ArgumentParser()
parser.add_argument(
    'pose_path', help='path to .npz for poses')
parser.add_argument(
    '--video',
    default=False,
    action='store_true',
    help='write video to path instead of displaying on screen')

if __name__ == '__main__':
    args = parser.parse_args()
    npz_path = args.pose_path
    print('Loading %s' % npz_path)
    loaded = np.load(npz_path)
    parents = loaded['parents']
    pose_keys = sorted([k for k in loaded.keys() if k.startswith('poses_')])
    print('Keys:', ', '.join(pose_keys))

    anims = []
    for key in pose_keys:
        this_anim = draw_poses(key, parents, loaded[key])
        anims.append((key, this_anim))

    if args.video:
        # save video
        print('Saving videos')
        with ThreadPoolExecutor() as p:
            # need thread pool because animations can't be picked for a
            # multiprocess pool :(
            for key in p.map(video_worker, anims):
                print('%s done' % key)
    else:
        print('Showing sequences')
        plt.show()
