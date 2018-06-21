#!/usr/bin/env python3
"""Trains an action predictor on estimated IkeaDB poses."""

import numpy as np

import addpaths  # noqa
from load import loadDataset
from common_pp.act_class_model import train_act_class_model


if __name__ == '__main__':
    dataset = loadDataset()
    merge_map = {
        'attach leg 1': '*tach',
        'attach leg 2': '*tach',
        'attach leg 3': '*tach',
        'attach leg 4': '*tach',
        'detach leg 1': '*tach',
        'detach leg 2': '*tach',
        'detach leg 3': '*tach',
        'detach leg 4': '*tach',
        'spin in': 'spin',
        'spin out': 'spin',
        'n/a': None
    }
    train_act_class_model(dataset, merge_map, balance_func=np.median)
