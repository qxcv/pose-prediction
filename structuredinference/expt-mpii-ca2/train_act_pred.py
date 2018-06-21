#!/usr/bin/env python3
"""Trains an action predictor on estimated MPII CA2 poses."""

import numpy as np

import addpaths  # noqa
from load import loadDataset
from common_pp.act_class_model import train_act_class_model


if __name__ == '__main__':
    dataset = loadDataset()
    merge_map = {
        'n/a': None
    }
    train_act_class_model(dataset, merge_map, balance_func=np.median)
