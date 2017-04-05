"""Utilities for loading skeletons in the NTU RGB+D format. Might work for
Kinect-recorded datasets, too."""

from collections import namedtuple
from enum import IntEnum

import numpy as np


class TrackState(IntEnum):
    """Kinect skeleton/joint tracking state. Details at
    https://goo.gl/Ayto0c"""
    # Position is garbage
    NOT_TRACKED = 0
    # Low confidence
    INFERRED = 1
    # High confidence
    TRACKED = 2


def str2bool(str):
    clean = str.strip().lower()
    if clean == 'true':
        return True
    if clean == 'false':
        return False
    try:
        return bool(int(clean))
    except ValueError:
        pass
    try:
        return bool(float(clean))
    except ValueError:
        pass
    raise ValueError('Could not interpret "%s" as a float' % str)


def str2track(str_ts):
    return TrackState(int(str_ts))


def fline(format, fp):
    """Extract a formatted line"""
    line_elems = next(fp).strip().split()
    format_flat = ''.join(format.strip().split())
    format_lookup = {'i': int, 'f': float, 'b': str2bool, 't': str2track}
    try:
        format_funcs = [format_lookup[c] for c in format_flat]
    except KeyError:
        raise ValueError('Invalid format %s (expect only symbols %s)' %
                         (format, ', '.join(format_lookup.keys())))
    if len(format) != len(line_elems):
        raise ValueError('Format expects %d item, but line has %d' %
                         (len(format), len(line_elems)))
    rv = [f(e) for f, e in zip(format_funcs, line_elems)]
    if len(rv) == 1:
        return rv[0]
    return rv


Skeleton = namedtuple('Skeleton', [
    'track_id', 'clip_flags', 'confidence_lh', 'state_lh', 'confidence_rh',
    'state_rh', 'restricted', 'skeleton', 'lean_x', 'lean_y', 'track_state'
])

joints_dtype = [
    ('x', float),
    ('y', float),
    ('z', float),
    ('depth_x', float),
    ('depth_y', float),
    ('colour_x', float),
    ('colour_y', float),
    ('orient_w', float),
    ('orient_x', float),
    ('orient_y', float),
    ('orient_z', float),
    ('track_state', int),
]


def load_skeletons(data):
    """Takes an iterable of lines and returns a loaded skeleton file."""
    num_frames = fline('i', data)
    frames = []
    for frame_num in range(num_frames):
        num_bodies = fline('i', data)
        bodies = []
        for body_num in range(num_bodies):
            # storing in a dict means I only have to write names out once :P
            d = {}
            (d['track_id'], d['clip_flags'], d['confidence_lh'], d['state_lh'],
             d['confidence_rh'], d['state_rh'], d['restricted'], d['lean_x'],
             d['lean_y'], d['track_state']) = fline('iififibfft', data)
            num_joints = fline('i', data)
            skelarray = np.recarray((num_joints, ), dtype=joints_dtype)
            for joint_num in range(num_joints):
                skelarray[joint_num] = np.asarray(
                    fline('f' * 11 + 'i', data), dtype='object')
            d['skeleton'] = skelarray
            bodies.append(Skeleton(**d))
        frames.append(bodies)
    return frames


Track = namedtuple('Track',
                   ['start_frame', 'end_frame', 'orig_id', 'skeletons'])


def _finalise_tracks(now_tracking, keys_to_remove, out_tracks, end_frame,
                     min_length):
    for key in sorted(keys_to_remove):
        start_frame, bodies = now_tracking[key]
        if end_frame - start_frame + 1 < min_length:
            continue
        skeletons = np.stack([b.skeleton for b in bodies], axis=0) \
                      .view(np.recarray)

        out_tracks.append(
            Track(
                start_frame=start_frame,
                end_frame=end_frame,
                orig_id=key,
                skeletons=skeletons))

        del now_tracking[key]


def extract_tracks(frames, min_length=1):
    """Turn per-frame skeleton representation into a list of frame tracks.
    Tracks start when an untracked skeleton appears, and end when the skeleton
    disapppears for one or more frames."""
    tracks = []
    now_tracking = {}
    num_discarded = 0
    for frame_num, frame in enumerate(frames):
        seen = set()
        for body in frame:
            # throw out skeletons which are not tracked as a whole, or which
            # have untracked joints
            joints_below = sum(joint.track_state < TrackState.TRACKED for
                               joint in body.skeleton)
            skel_below = body.track_state < TrackState.TRACKED
            if skel_below or joints_below > 0:
                num_discarded += 1
                # Debug prints
                # js = '%d untracked joints' % joints_below
                # st = 'person untracked' if skel_below else 'person tracked'
                # print('Throwing out skeleton (%s, %s)' % (st, js))
                continue

            tid = body.track_id
            seen.add(tid)
            # if we're tracking the person at the moment, add the new skeleton
            tid_tup = now_tracking.setdefault(tid, (frame_num, []))
            tid_tup[1].append(body)

        # create tracks for anyone out-of-frame
        to_remove = now_tracking.keys() - seen
        _finalise_tracks(now_tracking, to_remove, tracks, frame_num - 1,
                         min_length)

    # create tracks for any remaining people
    _finalise_tracks(now_tracking,
                     now_tracking.keys(), tracks, frame_num, min_length)

    return tracks, num_discarded
