#!/usr/bin/env python3

"""Tool for plotting sequences of generated 2D poses side-by-side."""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys


def draw_poses(title, parents, pose_sequence):
    N, T, _, J = pose_sequence.shape
    fig = plt.figure()
    fig.suptitle(title)

    # Arrange plots in a square
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / float(cols)))
    subplots = []
    all_lines = []
    for spi in range(N):
        ax = fig.add_subplot(rows, cols, spi+1)
        subplots.append(ax)

        sp_lines = []
        sp_joints = pose_sequence[spi]
        xmin, ymin = sp_joints.min(axis=0).min(axis=1)
        xmax, ymax = sp_joints.max(axis=0).max(axis=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.invert_yaxis()

        for joint in range(1, len(parents)):
            joint_coord = sp_joints[0, :, joint]
            parent = parents[joint]
            parent_coord = sp_joints[0, :, parent]
            x_data = (parent_coord[0], joint_coord[0])
            y_data = (parent_coord[1], joint_coord[1])
            line, = ax.plot(x_data, y_data)
            sp_lines.append(line)

        all_lines.append(sp_lines)

    def animate_step(frame, joints, parents, all_lines):
        redrawn = []

        for spi in range(len(all_lines)):
            sp_joints = joints[spi]
            lines = all_lines[spi]
            # one line per edge, so no line for the parent
            assert len(parents) == len(lines) + 1

            for joint in range(1, len(parents)):
                joint_coord = sp_joints[frame, :, joint]
                parent = parents[joint]
                parent_coord = sp_joints[frame, :, parent]
                line = lines[joint-1]
                line.set_xdata((parent_coord[0], joint_coord[0]))
                line.set_ydata((parent_coord[1], joint_coord[1]))
                redrawn.append(line)

        return redrawn

    return animation.FuncAnimation(fig, animate_step, frames=T, repeat=True,
                                   fargs=(pose_sequence, parents, all_lines),
                                   interval=1000*3/50.0, blit=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Need exactly one argument (.npz path for poses)"
    npz_path = sys.argv[1]
    print('Loading %s' % npz_path)
    loaded = np.load(npz_path)
    parents = loaded['parents']
    pose_keys = sorted([k for k in loaded.keys() if k.startswith('poses_')])
    print('Keys:', ','.join(pose_keys))

    anims = []
    for key in pose_keys:
        if 'train' in key:
            pass
        anim = draw_poses(key, parents, loaded[key])
        anims.append(anim)

    plt.show()