#!/usr/bin/env python2
"""Initialises paths to data loading scripts."""

import os
import sys

ppk_path = os.path.expanduser('~/repos/pose-prediction/keras')
pp_path = os.path.expanduser('~/repos/pose-prediction')
tm_path = os.path.expanduser('~/repos/theanomodels')
paths = {ppk_path, tm_path, pp_path}
for path in paths:
    assert os.path.isdir(path), 'code at %s must exist' % path
    if path not in sys.path:
        sys.path.append(path)

try:
    import stinfmodel_fast
except ImportError:
    # need to add to PATH
    this_dir = os.path.dirname(os.path.abspath(__file__))
    above = os.path.join(this_dir, '..')
    sys.path.append(above)

    # should not fail now that we've added it to PATH
    import stinfmodel_fast
