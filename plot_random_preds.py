#!/usr/bin/env python3
"""Takes a HDF5 file in the format accepted by stats_calculator and plots a few
sequences, chosen at random."""

from argparse import ArgumentParser

from plot_seqs import SequencePlotter2D

import sys
sys.path.append('keras/')
from p2d_loader import P2DDataset  # noqa: E402

parser = ArgumentParser()

if __name__ == '__main__':
    args = parser.parse_args()
