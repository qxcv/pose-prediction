#!/usr/bin/env python3
"""Pull poses out of a specified part of Ikea FA, then make three plots with
different (stupid) baselines."""

import argparse
import os
import re

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.io import loadmat
from scipy.misc import imread

# get_frame_paths and load_sorted_paths copied from structuredinference repo
# (oh well)
_num_re = re.compile(r'(\d+)')


def load_sorted_paths(frame_dir):
    """Sorts a bunch of paths that have numbers in the filename."""
    fns = os.listdir(frame_dir)
    everything = []
    for fn in fns:
        bn = os.path.basename(fn)
        # if we get many numbers then there is a bug
        num_str, = _num_re.findall(bn)
        thing_id = int(num_str)
        everything.append((thing_id, os.path.join(frame_dir, fn)))
    return [p for i, p in sorted(everything)]


def get_frame_paths(db_path, frame_dir, vid_name):
    db = loadmat(db_path, squeeze_me=True)['IkeaDB']
    # could get just one entry (one relevant to our vid) instead of looping
    # over all. oh well
    meta_dict = {}
    for video_entry in db:
        clip_path = video_entry['clip_path']
        prefix = '/data/home/cherian/IkeaDataset/Frames/'
        assert clip_path.startswith(prefix)
        path_suffix = clip_path[len(prefix):]
        # This is same number used to identify pose clip (not sequential!)
        tmp2_id = video_entry['video_id']
        new_name = 'vid%d' % tmp2_id
        meta_dict[new_name] = {'path_suffix': path_suffix, 'tmp2_id': tmp2_id}

    meta = meta_dict[vid_name]
    path_suffix = meta['path_suffix']
    tmp2_id = meta['tmp2_id']
    tmp2_id = int(re.match(r'^vid(\d+)$', vid_name).groups()[0])
    all_frame_fns = load_sorted_paths(os.path.join(frame_dir, path_suffix))
    # for some reason there is one video directory with a subdirectory that has
    # a numeric name
    frame_paths = [f for f in all_frame_fns if f.endswith('.jpg')]
    return frame_paths


def corrupt_wiener(poses, std):
    """Corrupt poses with simple Brownian motion."""
    # poses should be T*2*8
    # can consider making channels dependent if illustration is not dramatic
    # enough
    single_noise = std * np.random.randn(poses.shape[0], 2, 1)
    corr_noise = np.cumsum(single_noise, axis=0)
    return poses + corr_noise


def corrupt_gauss(poses, std):
    """Add Gaussian noise to each time step, independently."""
    single_noise = std * np.random.randn(*poses.shape)
    return poses + single_noise


def corrupt_flatline(poses):
    """Just predict the first real pose"""
    first_pose = poses[:1]
    return np.broadcast_to(first_pose, poses.shape)


def plot_pose(pose, parents, ax, line_spec):
    assert pose.shape == (2, 8)
    for child in range(1, len(parents)):
        parent = parents[child]
        line_top = pose[:, parent]
        line_bot = pose[:, child]
        ax.plot([line_bot[0], line_top[0]], [line_bot[1], line_top[1]],
                line_spec)


parser = argparse.ArgumentParser()
parser.add_argument('--sequence', default=None, help='name of sequence to use')
parser.add_argument(
    '--start-frame', type=int, default=0, help='first frame to use')
parser.add_argument(
    '--skip', type=int, default=15, help='frames to skip between')
parser.add_argument(
    '--tot-frames',
    type=int,
    default=8,
    help='total number of frames to produce')
parser.add_argument(
    '--dest-dir',
    default='comparison-images',
    help='where to put produced images')
parser.add_argument('--gauss-std', type=float, default=20)
parser.add_argument('--wiener-std', type=float, default=3)
parser.add_argument(
    '--frame-dir',
    default='/home/sam/sshfs/paloalto/data/home/cherian/IkeaDataset/Frames/',
    help='path to Ikea images')
parser.add_argument(
    '--db-path',
    default='/home/sam/sshfs/paloalto/data/home/cherian/IkeaDataset/IkeaClipsDB_withactions.mat',
    help='original IkeaDB')
parser.add_argument('poses', help='HDF5 file containing Ikea data')

if __name__ == '__main__':
    # LaTeX imitation
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'pgf.rcfonts': False,
        'pgf.texsystem': 'pdflatex',
        'xtick.labelsize': 'xx-small',
        'ytick.labelsize': 'xx-small',
        'legend.fontsize': 'xx-small',
        'axes.labelsize': 'x-small',
        'axes.titlesize': 'small',
    })

    print('Loading sequence')
    args = parser.parse_args()
    with h5py.File(args.poses, 'r') as fp:
        parents = fp['/parents'].value
        seq_choices = list(fp['/seqs'].keys())
        seq_name = args.sequence \
            or np.random.choice(seq_choices)
        poses = fp['/seqs/' + seq_name + '/poses'].value

    print("Loading frame paths for sequence '%s'" % seq_name)
    frame_paths = get_frame_paths(args.db_path, args.frame_dir, seq_name)

    start = args.start_frame
    stop = start + args.tot_frames * args.skip
    subseq = poses[start:stop]
    # I was going to multiply noise deviations by this, but that's a bad idea
    # base_std = np.std(poses, axis=0).flatten().mean()
    orig_spec = 'g-'
    corrupted = [# ('Constant', '^c--', corrupt_flatline(subseq)),
                 ('Gaussian', '+m--', corrupt_gauss(subseq, args.gauss_std)),
                 ('Wiener', '*y--', corrupt_wiener(subseq, args.wiener_std))]

    dest_dir = os.path.join(args.dest_dir, '%s-from-%i-to-%i-skip-%i' %
                            (seq_name, start, stop, args.skip))
    os.makedirs(dest_dir, exist_ok=True)

    print('Making frame snapshots')
    figs = []
    for subseq_ind in range(0, stop - start, args.skip):
        ss_fig = plt.figure(frameon=False)
        ss_ax = ss_fig.gca()

        # plot image
        frame_path = frame_paths[subseq_ind + start]
        frame = imread(frame_path)
        ss_ax.imshow(frame)

        # plot poses (not original)
        orig_pose = subseq[subseq_ind]
        # plot_pose(orig_pose, parents, ss_ax, orig_spec)
        for name, line_spec, corrupt_subseq in corrupted:
            corrupt_pose = corrupt_subseq[subseq_ind]
            plot_pose(corrupt_pose, parents, ss_ax, line_spec)
        # centre around the original pose
        pose_width, pose_height \
            = np.max(orig_pose, axis=1) - np.min(orig_pose, axis=1)
        box_side = 1.2 * max(pose_width, pose_height)
        box_center = np.mean(orig_pose, axis=1)
        box_xmin, box_ymin = box_center - box_side / 2.0
        box_xmax, box_ymax = box_center + box_side / 2.0
        ss_ax.set_xlim(xmin=box_xmin, xmax=box_xmax)
        # flip ys to make sure image is right way up
        ss_ax.set_ylim(ymax=box_ymin, ymin=box_ymax)
        # ss_ax.axis('tight')
        # ss_ax.axis('image')
        ss_ax.axis('off')
        ss_ax.xaxis.set_major_locator(ticker.NullLocator())
        ss_ax.yaxis.set_major_locator(ticker.NullLocator())
        ss_fig.savefig(
            os.path.join(dest_dir, 'frame-%d.pdf' % subseq_ind),
            bbox_inches='tight',
            pad_inches=0,
            transparent=True)

    print('Plotting coordinate error')
    err_fig = plt.figure()
    err_ax = err_fig.gca()
    times = np.arange(subseq.shape[0])
    for name, line_spec, corrupt_seq in corrupted:
        # mean joint position error
        dists = np.linalg.norm(corrupt_seq - subseq, axis=1).mean(axis=-1)
        err_ax.plot(times, dists, line_spec)
    err_ax.set_xlabel('Frames')
    err_ax.set_ylabel('Mean joint error')
    err_ax.set_xlim(xmin=min(times), xmax=max(times))
    err_ax.set_ylim(ymin=0)

    err_fig.set_size_inches((3.25, 3.25))
    err_fig.tight_layout()
    err_fig.savefig(
        os.path.join(dest_dir, 'accuracy.pdf'),
        bbox_inches='tight',
        transparent=True)
