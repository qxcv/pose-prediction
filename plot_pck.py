#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import cycle
from os import path

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
    '--method-names',
    nargs='+',
    default=None,
    help='"Pretty" names for methods')
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
parser.add_argument(
    '--xtype',
    choices=['thresh', 'time'],
    default='thresh',
    help='choice of dimension for x-axis')
parser.add_argument(
    '--fps',
    type=float,
    default=None,
    help="Frames-per-second (converts from frame numbers to times)")


def load_data(directory, method_names, part_names):
    all_thresh = None
    all_times = None
    data_table = {}
    for method in method_names:
        for part in part_names:
            csv_path = path.join(directory, 'pck_%s_%s.csv' % (method, part))
            data = np.loadtxt(csv_path, delimiter=',')
            times = data[:, 0].astype(int)
            # pcks[time, thresh] gives PCK at given threshold and time step (in
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


def select_thresh_ind(data_table, parts, thresholds, method1, method2):
    """Select threshold that best maximises data separation between two
    methods."""
    costs = np.zeros_like(thresholds)
    for part in parts:
        # pcks[thresh,time] gives PCK at given threshold and time step (in
        # [0, 1])
        meth1_pcks = data_table[(method1, part)]
        meth2_pcks = data_table[(method2, part)]
        separations = np.sum(meth1_pcks - meth2_pcks, axis=0)
        costs[:] += separations
    return np.argmax(costs)


def plot_xtype_thresh(data_table, all_thresholds, all_times, method_labels,
                      args):
    methods = args.methods
    parts = args.parts
    sel_times = np.array(list(map(int, args.times)))

    # Time goes vertically downwards, parts go left-to-right
    _, subplots = plt.subplots(
        len(sel_times), len(parts), sharey=True, sharex=True)
    common_handles = None
    for col, part in enumerate(parts):
        for row, time in enumerate(sel_times):
            subplot = subplots[row][col]
            pcks = []
            for method in methods:
                method_pcks = data_table[(method, part)][time, :]
                pcks.append(method_pcks)
            if common_handles is None:
                # Record first lot of handles for reuse
                common_handles = []
                for pck, label, marker in zip(pcks, methods, cycle(MARKERS)):
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
            pl = 'frames' if time != 1 else 'frame'
            subplot.set_title('%s after %d %s' % (part.title(), time, pl))
            is_last_row = row == len(sel_times) - 1
            if is_last_row:
                if args.thresh_is_px:
                    subplot.set_xlabel('Threshold (px)')
                else:
                    subplot.set_xlabel('Threshold')
            subplot.grid(which='both')

            if args.xmax is not None:
                subplot.set_xlim(xmax=args.xmax)

    return subplots[0][0], common_handles


def plot_xtype_time(data_table, all_thresholds, all_times, method_labels,
                    args):
    methods = args.methods
    parts = args.parts
    thresh_ind = select_thresh_ind(data_table, parts, all_thresholds,
                                   methods[0], methods[1])
    threshold = all_thresholds[thresh_ind]
    if args.fps is None:
        x_axis = all_times
    else:
        x_axis = all_times / float(args.fps)

    # Parts go left-to-right, there are no times
    _, subplots = plt.subplots(1, len(parts), sharey=True, sharex=True)
    common_handles = None
    for col, part in enumerate(parts):
        subplot = subplots[col]
        pcks = []
        for method in methods:
            pckt = data_table[(method, part)]
            assert pckt.shape == all_times.shape + all_thresholds.shape
            method_pcks = pckt[:, thresh_ind]
            pcks.append(method_pcks)
        if common_handles is None:
            # Record first lot of handles for reuse
            common_handles = []
            for pck, label, marker in zip(pcks, method_labels, cycle(MARKERS)):
                handle, = subplot.plot(
                    x_axis, 100 * pck, label=label, marker=marker)
                common_handles.append(handle)
        else:
            for pck, handle in zip(pcks, common_handles):
                props = handle.properties()
                kwargs = {k: v for k, v in props.items() if k in COMMON_PROPS}
                subplot.plot(x_axis, 100 * pck, **kwargs)

        # Labels, titles
        subplot.set_title('{}@{:.2g}'.format(part.title(), threshold))
        subplot.grid(which='both')
        if args.fps is None:
            subplot.set_xlabel('Frame number')
        else:
            subplot.set_xlabel('Time (s)')

        if args.xmax is not None:
            subplot.set_xlim(xmax=args.xmax)

    return subplots[0], common_handles


if __name__ == '__main__':
    args = parser.parse_args()

    # Make the plot look like LaTeX
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

    # Load stuff from CSVs
    data_table, all_thresholds, all_times = load_data(args.stats_dir,
                                                      args.methods, args.parts)
    if args.method_names is None:
        method_labels = args.methods
    else:
        # the 10 spaces are in there so that MPL doesn't cut off my plot (yeah,
        # yeah, you try to fix it "the right way" smartass)
        method_labels = [n + ' ' * 10 for n in args.method_names]
        assert len(method_labels) == len(args.methods), \
            "'Pretty' method names should match with ordinary ones"

    # Plot the data in whatever arrangement the user asked for
    if args.xtype == 'thresh':
        sp_leg, handles = plot_xtype_thresh(data_table, all_thresholds,
                                            all_times, method_labels, args)
    elif args.xtype == 'time':
        sp_leg, handles = plot_xtype_time(data_table, all_thresholds,
                                          all_times, method_labels, args)
    else:
        raise ValueError('Unknown x-axis type %s' % args.xtype)

    # Deal with legend, y-axis
    master_ax = plt.gcf().add_subplot(1, 1, 1)
    for side in ['top', 'bottom', 'left', 'right']:
        master_ax.spines[side].set_color('none')
    master_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                          right='off')
    master_ax.set_ylabel('Accuracy (%)')
    sp_leg.set_ylim(ymin=0, ymax=100)
    minor_locator = AutoMinorLocator(2)
    sp_leg.yaxis.set_minor_locator(minor_locator)
    sp_leg.set_yticks(range(0, 101, 20))
    # ax = plt.gca()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.2, box.height])
    # if args.legend_below:
    #     bbox = (0.05, -0.01, 0.9, 0.1)
    # else:
    #     bbox = (0.05, 0.88, 0.9, 0.1)
    legend = plt.figlegend(
        handles,
        method_labels,
        bbox_to_anchor=(1.15, 0.5),
        loc="right",
        frameon=False
    )

    # Save or show
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
