#!/usr/bin/env python3
"""Tool for plotting sequences of generated 2D poses side-by-side."""

import argparse
from concurrent.futures import ThreadPoolExecutor

from matplotlib import animation as anim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
from scipy.misc import imread

from expmap import expmap_to_xyz, plot_xyz_skeleton


class SequencePlotter(object):
    """Base class for animating 2D or 3D poses in a grid."""

    def __init__(self,
                 title,
                 sequences,
                 action_labels=None,
                 subplot_titles=None,
                 fps=15,
                 crossover=None):
        self.crossover = crossover
        self.fps = fps
        self.subplot_titles = subplot_titles
        self.sequences = sequences
        self.N, self.T = self.sequences.shape[:2]
        self.fig = plt.figure()
        self.fig.suptitle(title)

        # Arrange plots in a square
        cols = int(np.ceil(np.sqrt(self.N)))
        rows = int(np.ceil(self.N / float(cols)))
        self.subplots = []
        self.action_labels = action_labels
        if self.action_labels is not None:
            self.label_handles = []
        for spi in range(self.N):
            if self.is_3d:
                ax = self.fig.add_subplot(rows, cols, spi + 1, projection='3d')
            else:
                ax = self.fig.add_subplot(rows, cols, spi + 1)

            self.subplots.append(ax)

            if self.action_labels is not None:
                # plot action names (if any) above subplots
                label = self.action_labels[0]
                if not label:
                    label = ''
                text_h = ax.text(0, -10, 'Action: ' + label)
                self.label_handles.append(text_h)

            if self.subplot_titles is not None:
                ax.set_title(self.subplot_titles[spi])

    def make_anim(self):
        self.animation = anim.FuncAnimation(
            self.fig,
            self.animate_step,
            frames=self.T,
            repeat=True,
            interval=1000 * (1 / self.fps),
            blit=False)

    def animate_step(self, frame_number):
        redrawn = []

        for spi in range(self.N):
            ax = self.subplots[spi]
            redrawn.extend(self.draw_frame(ax, spi, frame_number))

            if self.action_labels is not None:
                label = self.action_labels[frame_number]
                if not label:
                    label = ''
                handle = self.label_handles[spi]
                new_text = 'Action: ' + label
                if handle.get_text() != new_text:
                    handle.set_text(new_text)
                    redrawn.append(handle)

            if self.crossover is not None:
                if frame_number >= self.crossover:
                    colour = 'blue'
                else:
                    colour = 'green'
                ax = self.subplots[spi]
                spines = list(ax.spines.values())
                for spine in spines:
                    spine.set_linewidth(6)
                    spine.set_color(colour)
                    redrawn.append(spine)

        return redrawn

    def setup(self):
        raise NotImplementedError()

    def draw_frame(self, ax, sequence_id, frame_number):
        raise NotImplementedError()


class SequencePlotter2D(SequencePlotter):
    is_3d = False

    def __init__(self,
                 *args,
                 parents=None,
                 frame_paths=None,
                 frames=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert frame_paths is None or frames is None, \
            "can take frame paths or loaded frames, but not both"
        self.frame_paths = frame_paths
        self.frames = frames
        # for redraw caching
        self.mpl_handles = {}

        assert parents is not None, "Need parents to draw poses!"
        self.parents = parents

        self.make_anim()

    def get_frame(self, sequence_id, frame_number):
        # get a frame for the given sequence and frame, if possible, or return
        # None
        if self.frames is not None:
            return self.frames[sequence_id][frame_number]
        elif self.frame_paths is not None:
            return imread(self.frame_paths[sequence_id][frame_number])

    def draw_first(self, ax, sequence_id):
        handles = self.mpl_handles.setdefault(sequence_id, {})
        poses = self.sequences[sequence_id]
        pose = poses[0]
        frame = self.get_frame(sequence_id, 0)
        if frame is None:
            # if no image, set plot boundaries
            xmin, ymin = poses.min(axis=0).min(axis=1)
            xmax, ymax = poses.max(axis=0).max(axis=1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.invert_yaxis()
        else:
            # if image, it implicitly defines boundaries
            handles['image'] = ax.imshow(frame)

        lines = handles['lines'] = []
        for joint in range(1, len(self.parents)):
            # plot actual skeleton
            joint_coord = pose[:, joint]
            parent = self.parents[joint]
            parent_coord = pose[:, parent]
            x_data = (parent_coord[0], joint_coord[0])
            y_data = (parent_coord[1], joint_coord[1])
            line, = ax.plot(x_data, y_data)
            lines.append(line)

        if frame is not None:
            # if there's an image frame then we don't want to see anything else
            # except the image and the pose (no ticks, no blank padding, etc.)
            # I expected ax.imshow() would handle this, but I guess not
            ax.set_axis_off()
            ax.axis('image')
            ax.set_xlim([0, frame.shape[1]])
            ax.set_ylim([0, frame.shape[0]])
            ax.invert_yaxis()

    def draw_subsequent(self, sequence_id, frame_number):
        pose = self.sequences[sequence_id][frame_number]
        handles = self.mpl_handles[sequence_id]
        lines = handles['lines']
        redrawn = []
        # one line per edge, so no line for the parent
        assert len(self.parents) == len(lines) + 1

        frame = self.get_frame(sequence_id, frame_number)
        if frame is not None:
            handle = handles['image']
            handle.set_data(frame)
            redrawn.append(handle)

        for joint in range(1, len(self.parents)):
            joint_coord = pose[:, joint]
            parent = self.parents[joint]
            parent_coord = pose[:, parent]
            line = lines[joint - 1]
            line.set_xdata((parent_coord[0], joint_coord[0]))
            line.set_ydata((parent_coord[1], joint_coord[1]))
            redrawn.append(line)

    def draw_frame(self, ax, sequence_id, frame_number):
        if sequence_id not in self.mpl_handles:
            # first frame needs to set up handles
            self.draw_first(ax, sequence_id)
        else:
            # we have handles, all good!
            self.draw_subsequent(sequence_id, frame_number)

        return [h for l in self.mpl_handles.values() for h in l]


class SequencePlotter3D(SequencePlotter):
    """Plots several sequences of 3D skeletons in (x,y,z) parameterisation."""

    is_3d = True

    def __init__(self, *args, parents=None, bone_lengths=None, **kwargs):
        super().__init__(*args, **kwargs)

        # for redraw caching
        self.mpl_handles = {}

        self.parents = parents
        self.bone_lengths = bone_lengths
        self.xyz_skels = []
        for seq in self.sequences:
            xyz = expmap_to_xyz(seq, self.parents, self.bone_lengths)
            self.xyz_skels.append(xyz)

        self.make_anim()

    def draw_frame(self, ax, sequence_id, frame_number):
        xyz_skeleton = self.xyz_skels[sequence_id][frame_number]
        rv = self.mpl_handles[sequence_id] = plot_xyz_skeleton(
            xyz_skeleton, self.parents, ax, self.mpl_handles.get(sequence_id))
        return rv


def draw_poses(title,
               parents,
               pose_sequences,
               frame_paths=None,
               frames=None,
               subplot_titles=None,
               fps=50 / 3.0,
               crossover=None,
               action_labels=None):
    """Deprecated function to plot 2D sequence. Use SequencePlotter2D directly,
    instead."""
    plotter = SequencePlotter2D(
        title,
        pose_sequences,
        parents=parents,
        frame_paths=frame_paths,
        frames=frames,
        subplot_titles=subplot_titles,
        fps=fps,
        crossover=crossover,
        action_labels=action_labels)
    return plotter.animation


def video_worker(ks):
    key, plotter = ks
    print('Working on', key)
    plotter.animation.save(key + '.mp4', writer='avconv', fps=plotter.fps)
    return key


parser = argparse.ArgumentParser()
parser.add_argument('pose_path', help='path to .npz for poses')
parser.add_argument(
    '--video',
    default=False,
    action='store_true',
    help='write video to path instead of displaying on screen')
parser.add_argument(
    '--3d',
    dest='is_3d',
    default=False,
    action='store_true',
    help='look for 3D sequences instead of 2D ones')

if __name__ == '__main__':
    args = parser.parse_args()
    npz_path = args.pose_path
    print('Loading %s' % npz_path)
    loaded = np.load(npz_path)
    is_3d = args.is_3d
    # 2D sequences are stored in .npz as poses_<name>, while 3D sequences are
    # stored as skeletons_<name>
    if not is_3d:
        parents = loaded['parents']
        data_keys = sorted(
            [k for k in loaded.keys() if k.startswith('poses_')])
    else:
        # assume expmap parameterisation
        parents_3d = loaded['parents_3d']
        bone_lengths = loaded['bone_lengths_3d']
        data_keys = sorted(
            [k for k in loaded.keys() if k.startswith('skeletons_')])
    print('Keys:', ', '.join(data_keys))

    plotters = []
    for key in data_keys:
        if is_3d:
            plotter = SequencePlotter3D(
                key,
                loaded[key],
                parents=parents_3d,
                bone_lengths=bone_lengths)
        else:
            plotter = SequencePlotter2D(key, loaded[key], parents=parents)
        plotters.append((key, plotter))

    if args.video:
        # save video
        print('Saving videos')
        with ThreadPoolExecutor() as p:
            # need thread pool because animations can't be picked for a
            # multiprocess pool :(
            for key in p.map(video_worker, plotters):
                print('%s done' % key)
    else:
        print('Showing sequences')
        plt.show()
