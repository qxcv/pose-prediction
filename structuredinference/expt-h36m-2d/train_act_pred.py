#!/usr/bin/env python3
"""Trains an action predictor on estimated H3.6M 2D poses."""

import numpy as np

import addpaths  # noqa
from load import loadDataset
from common_pp.act_class_model import train_act_class_model


if __name__ == '__main__':
    dataset = loadDataset()
    merge_map = {}
    train_act_class_model(dataset, merge_map, balance_func=np.median)
