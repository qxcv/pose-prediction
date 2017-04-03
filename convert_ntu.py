#!/usr/bin/env python3
"""Convert NTU RGB+D dataset (2D and 3D)."""

import argparse
from json import dumps
from os import path
import re
from zipfile import ZipFile

import h5py
import numpy as np
from tqdm import tqdm

from ntu import (ACTION_CLASSES, BAD_IDENTIFIERS, load_skeletons,
                 extract_tracks, JOINT_NAMES)

# From readme in example code repo: SsssCcccPpppRrrrAaaa. S, C, P, R and A
# denote setup, camera ID, performer ID, replication number (1/2), action
# label, respectively.
_ident_re = re.compile(r'^S(?P<setup>\d{3})C(?P<camera>\d{3})'
                       r'P(?P<performer>\d{3})R(?P<replication>\d{3})'
                       r'A(?P<action>\d{3})$')


def parse_ident(ident):
    """Extracts metadata from NTU RGB+D filenames."""
    match = _ident_re.match(ident)
    if match is None:
        raise ValueError("Couldn't parse NTU-RGB+D identifier '%s'" % ident)
    gd = match.groupdict()
    rv = {k: int(v) for k, v in gd.items()}
    rv['ident'] = ident
    return rv


def read_skeleton_file(zip_file, path_to_skeleton):
    # parse filename
    basename = path.basename(path_to_skeleton)
    ext = '.skeleton'
    if not basename.endswith(ext):
        raise ValueError('"%s" does not end with "%s", is it a skeleton?' %
                         (path_to_skeleton, ext))
    ident = basename[:-len(ext)]
    skelly_meta = parse_ident(ident)

    # load_skeletons
    line_iter = iter(zip_file.read(path_to_skeleton).splitlines())
    skelly_frames = load_skeletons(line_iter)
    skelly_tracks = extract_tracks(skelly_frames)

    return skelly_meta, skelly_tracks


JNI = {name: i for i, name in enumerate(JOINT_NAMES)}
RHIP_ID = JNI['HipRight']
LHIP_ID = JNI['HipLeft']
LSHOL_ID = JNI['ShoulderLeft']
RSHOL_ID = JNI['ShoulderRight']
JOINTS_TO_KEEP = [
    'Head', 'Neck', 'ShoulderRight', 'ElbowRight', 'WristRight',
    'ShoulderLeft', 'ElbowLeft', 'WristLeft'
]
TO_KEEP_INDS = list(map(lambda k: JNI[k], JOINTS_TO_KEEP))


def rescale(skeletons_2d):
    # want to make sure we haven't got integer data
    skeletons_2d = skeletons_2d.astype(float)
    fst_dists = np.linalg.norm(
        skeletons_2d[..., RHIP_ID] - skeletons_2d[..., LSHOL_ID], axis=1)
    snd_dists = np.linalg.norm(
        skeletons_2d[..., LHIP_ID] - skeletons_2d[..., RSHOL_ID], axis=1)
    scale = np.median(np.concatenate([fst_dists, snd_dists]))
    # TODO: fix scale thing for small items
    if scale <= 1e-5:
        scale = 1

    # Corresponds to Top of head, Neck, Right shoulder, Right elbow, Right
    # wrist, Left shoulder, Left elbow, Left wrist (CPM)
    trimmed_skeleton = skeletons_2d[:, :, TO_KEEP_INDS]
    return trimmed_skeleton / scale, scale


parser = argparse.ArgumentParser()
parser.add_argument('ntu_path', help='path to NTU RGB+D skeleton zip file')
parser.add_argument('dest', help='path for HDF5 output file')

if __name__ == '__main__':
    args = parser.parse_args()
    with ZipFile(args.ntu_path) as in_fp, h5py.File(args.dest) as out_fp:
        for seq_path in tqdm(in_fp.namelist(), smoothing=0.005):
            if not seq_path.endswith('.skeleton'):
                continue
            skelly_meta, skelly_tracks = read_skeleton_file(in_fp, seq_path)
            ident = skelly_meta['ident']
            if ident in BAD_IDENTIFIERS:
                continue
            for track in skelly_tracks:
                # need to make sure we can recover whatever this track was in
                # the original files
                track_name = ('{ident}_I{orig_id:03d}SF{start_frame:05d}'
                              'EF{end_frame:05d}') \
                             .format(ident=ident,
                                     orig_id=track.orig_id,
                                     start_frame=track.start_frame,
                                     end_frame=track.end_frame)
                prefix = '/seqs/' + track_name + '/'

                action_id = skelly_meta['action']
                prefix_2d = '/seqs/' + track_name + '/'
                # Confusing terminology, but "depth_x" = "x coordinate of joint
                # in depth image"
                skeleton_xy = np.stack(
                    [track.skeletons.depth_x, track.skeletons.depth_y], axis=1)
                scaled_skels, scale = rescale(skeleton_xy)
                num_frames = len(skeleton_xy)
                out_fp[prefix + 'poses'] = skeleton_xy.astype('float32')
                out_fp[prefix + 'actions'] \
                    = np.full((num_frames, ), action_id).astype('uint8')
                out_fp[prefix + 'valid'] = np.ones_like(skeleton_xy) \
                                             .astype('uint8')
                out_fp[prefix + 'scale'] = scale

                # same, but for 3D data
                prefix_3d = '/seqs3d/' + track_name + '/'
                # TODO

        action_names = [n for i, n in sorted(ACTION_CLASSES.items())]
        # use CPM parents
        out_fp['/parents'] = np.array([0, 0, 1, 2, 3, 1, 5, 6], dtype=int)
        out_fp['/action_names'] = np.array(
            [ord(c) for c in dumps(action_names)], dtype='uint8')
        out_fp['/num_actions'] = len(action_names)
