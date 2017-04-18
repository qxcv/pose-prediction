#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import cycle
from os import path
from sys import exit, stderr

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

THRESH = 'Threshold'
MARKERS = [None]
# List of matplotlib.artist.Artist properties which we should copy between
# subplots
COMMON_PROPS = {
    'linestyle', 'marker', 'visible', 'drawstyle', 'linewidth',
    'markeredgewidth', 'markeredgecolor', 'markerfacecoloralt',
    'dash_joinstyle', 'zorder', 'markersize', 'solid_capstyle',
    'dash_capstyle', 'markevery', 'fillstyle', 'markerfacecolor', 'label',
    'alpha', 'path_effects', 'color', 'solid_joinstyle'
}

parser = ArgumentParser(
    description="Take accuracy at different thresholds and plot it nicely")
parser.add_argument(
    '--save',
    metavar='PATH',
    type=str,
    default=None,
    help="Destination file for graph")
parser.add_argument('--stats-dir', help='directory in which .csvs are located')
parser.add_argument(
    '--methods',
    nargs='+',
    # action='append',
    default=[],
    help='Add an algorithm to the plot, by name')
parser.add_argument(
    '--parts',
    nargs='+',
    # action='append',
    default=[],
    help='Add a joint to the plot, by name')
parser.add_argument(
    '--times',
    nargs='+',
    # action='append',
    default=[],
    help='Add a time to the plot')
parser.add_argument(
    '--xmax', type=float, default=None, help='Maximum value along the x-axis')
parser.add_argument(
    '--no-thresh-px',
    action='store_false',
    dest='thresh_is_px',
    default=True,
    help='Disable (px) annotation for threshold')
parser.add_argument(
    '--legend-below',
    action='store_true',
    dest='legend_below',
    default=False,
    help='Put the legend below the plot rather than above it')
parser.add_argument(
    '--dims',
    nargs=2,
    type=float,
    metavar=('WIDTH', 'HEIGHT'),
    default=[6, 3],
    help="Dimensions (in inches) for saved plot")


def load_data(directory, method_names, part_names):
    all_thresh = None
    all_times = None
    data_table = {}
    for method in method_names:
        for part in part_names:
            csv_path = path.join(directory, 'pck_%s_%s.csv' % (method, part))
            data = np.loadtxt(csv_path, delimiter=',')
            times = data[:, 0].astype(int)
            # pcks[thresh,time] gives PCK at given threshold and time step (in
            # [0, 1])
            pcks = data[:, 1:]
            with open(csv_path, 'r') as fp:
                first_line = next(fp).strip()
            cols = first_line.split(',')[1:]
            thresholds = np.array([float(c.lstrip('@')) for c in cols])

            # make sure times and thresholds match for each point
            if all_times is None:
                all_times = times
            assert np.all(all_times == times)
            if all_thresh is None:
                all_thresh = thresholds
            assert np.all(all_thresh == thresholds)

            data_table[(method, part)] = pcks

    return data_table, all_thresh, all_times


if __name__ == '__main__':
    args = parser.parse_args()

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

    methods = args.methods
    labels = methods
    parts = args.parts
    sel_times = np.array(list(map(int, args.times)))
    data_table, all_thresholds, all_times = load_data(args.stats_dir, methods,
                                                      parts)
    sel_time_inds = [np.nonzero(all_times == t)[0] for t in sel_times]

    # Time goes vertically downwards, parts go left-to-right
    _, subplots = plt.subplots(
        len(sel_times), len(parts), sharey=True, sharex=True)
    common_handles = None
    for col, part in enumerate(parts):
        for row, time in enumerate(sel_times):
            sel_time_ind = sel_time_inds[row]
            subplot = subplots[row][col]
            pcks = []
            for method in methods:
                method_pcks = data_table[(method, part)][time, :]
                pcks.append(method_pcks)
            if common_handles is None:
                # Record first lot of handles for reuse
                common_handles = []
                for pck, label, marker in zip(pcks, labels, cycle(MARKERS)):
                    handle, = subplot.plot(
                        all_thresholds, 100 * pck, label=label, marker=marker)
                    common_handles.append(handle)
            else:
                for pck, handle in zip(pcks, common_handles):
                    props = handle.properties()
                    kwargs = {
                        k: v
                        for k, v in props.items() if k in COMMON_PROPS
                    }
                    subplot.plot(all_thresholds, 100 * pck, **kwargs)

            # Labels, titles
            subplot.set_title('%s after %d frames' % (part, time))
            if args.thresh_is_px:
                subplot.set_xlabel('Threshold (px)')
            else:
                subplot.set_xlabel('Threshold')
            subplot.grid(which='both')

            if args.xmax is not None:
                subplot.set_xlim(xmax=args.xmax)

    subplots[0][0].set_ylabel('Accuracy (%)')
    subplots[0][0].set_ylim(ymin=0, ymax=100)
    minor_locator = AutoMinorLocator(2)
    subplots[0][0].yaxis.set_minor_locator(minor_locator)
    subplots[0][0].set_yticks(range(0, 101, 20))
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if args.legend_below:
        bbox = (0.05, 0, 0.9, 0.1)
    else:
        bbox = (0.05, 0.88, 0.9, 0.1)
    legend = plt.figlegend(
        common_handles,
        labels,
        bbox_to_anchor=bbox,
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0,
        frameon=False)

    if args.save is None:
        plt.show()
    else:
        print('Saving figure to', args.save)
        plt.gcf().set_size_inches(args.dims)
        plt.tight_layout()
        plt.savefig(
            args.save,
            bbox_inches='tight',
            bbox_extra_artists=[legend],
            transparent=True)
