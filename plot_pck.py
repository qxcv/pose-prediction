#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import cycle
from os import path
from sys import exit, stderr

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pandas import read_csv

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
PRIORITIES = {
    'Shoulders': 0,
    'Elbows': 1,
    'Wrists': 2
}

parser = ArgumentParser(
    description="Take accuracy at different thresholds and plot it nicely"
)
parser.add_argument(
    '--save', metavar='PATH', type=str, default=None,
    help="Destination file for graph"
)
parser.add_argument(
    '--input', nargs=2, metavar=('NAME', 'PATH'), action='append', default=[],
    help='Name (title) and path of CSV to plot; can be specified repeatedly'
)
parser.add_argument(
    '--poster', action='store_true', dest='is_poster', default=False,
    help='Produce a plot for the poster rather than the report.'
)
parser.add_argument(
    '--xmax', type=float, default=None, help='Maximum value along the x-axis'
)
parser.add_argument(
    '--colnames', type=str, nargs='*', default=None, help='Names of columns to use'
)
parser.add_argument(
    '--no-thresh-px', action='store_false', dest='thresh_is_px', default=True,
    help='Disable (px) annotation for threshold'
)
parser.add_argument(
    '--legend-below', action='store_true', dest='legend_below', default=False,
    help='Put the legend below the plot rather than above it'
)
parser.add_argument(
    '--dims', nargs=2, type=float, metavar=('WIDTH', 'HEIGHT'),
    default=[6, 3], help="Dimensions (in inches) for saved plot"
)


def load_data(inputs, part_names=None):
    labels = []
    all_thresholds = []
    if part_names is not None:
        parts = {part: [] for part in part_names}
    else:
        parts = None

    for name, path in inputs:
        labels.append(name)
        csv = read_csv(path)
        thresholds = csv[THRESH]
        all_thresholds.append(thresholds)

        if parts is None:
            parts = {part: [] for part in csv.columns.difference([THRESH])}

        for part in parts:
            part_vals = csv[part]
            assert len(part_vals) == len(thresholds)
            parts[part].append(part_vals)

    return labels, all_thresholds, parts

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.input:
        parser.print_usage(stderr)
        print('error: must specify at least one --input', file=stderr)
        exit(1)

    if args.is_poster:
        matplotlib.rcParams.update({
            'font.family': 'Ubuntu',
            'pgf.rcfonts': False,
            'xtick.labelsize': '12',
            'ytick.labelsize': '12',
            'legend.fontsize': '14',
            'axes.labelsize': '16',
            'axes.titlesize': '18',
        })
    else:
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

    labels, all_thresholds, parts = load_data(args.input, args.colnames)

    _, subplots = plt.subplots(1, len(parts), sharey=True)
    common_handles = None
    part_keys = sorted(parts.keys(), key=lambda s: PRIORITIES.get(s, -1))
    for part_name, subplot in zip(part_keys, subplots):
        pcks = parts[part_name]
        if common_handles is None:
            # Record first lot of handles for reuse
            common_handles = []
            for pck, label, thresholds, marker in zip(pcks, labels,
                    all_thresholds, cycle(MARKERS)):
                handle, = subplot.plot(
                    thresholds, 100 * pck, label=label, marker=marker
                )
                common_handles.append(handle)
        else:
            for pck, thresholds, handle in zip(pcks, all_thresholds, common_handles):
                props = handle.properties()
                kwargs = {k: v for k, v in props.items() if k in COMMON_PROPS}
                subplot.plot(thresholds, 100 * pck, **kwargs)

        # Labels, titles
        subplot.set_title(part_name)
        if args.thresh_is_px:
            subplot.set_xlabel('Threshold (px)')
        else:
            subplot.set_xlabel('Threshold')
        subplot.grid(which='both')

        if args.xmax is not None:
            subplot.set_xlim(xmax=args.xmax)

    subplots[0].set_ylabel('Accuracy (%)')
    subplots[0].set_ylim(ymin=0, ymax=100)
    minor_locator = AutoMinorLocator(2)
    subplots[0].yaxis.set_minor_locator(minor_locator)
    subplots[0].set_yticks(range(0, 101, 20))
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if args.is_poster:
        legend = plt.figlegend(
            common_handles, labels, 'center left', bbox_to_anchor=(0.965, 0.5),
            borderaxespad=0.
        )
    else:
        if args.legend_below:
            bbox = (0.05, 0, 0.9, 0.1)
        else:
            bbox = (0.05, 0.88, 0.9, 0.1)
        legend = plt.figlegend(
            common_handles, labels, bbox_to_anchor=bbox,
            loc=3, ncol=3, mode="expand", borderaxespad=0,
            frameon=False
        )

    if args.save is None:
        plt.show()
    else:
        print('Saving figure to', args.save)
        plt.gcf().set_size_inches(args.dims)
        plt.tight_layout()
        plt.savefig(
            args.save, bbox_inches='tight', bbox_extra_artists=[legend],
            transparent=True
        )
