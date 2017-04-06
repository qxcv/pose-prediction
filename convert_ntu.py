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
                 extract_tracks, JOINT_NAMES, JOINT_PARENT_INDS,
                 EVAL_PERFORMERS)
from expmap import xyz_to_expmap, bone_lengths

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
    skelly_tracks, discarded = extract_tracks(skelly_frames)

    return skelly_meta, skelly_tracks, discarded


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


def h5_json_encode(data):
    """Turns rich Python data structure into array of bytes so that it can be
    stuffed in an HDF5 file."""
    char_codes = [ord(c) for c in dumps(data)]
    return np.array(char_codes, dtype='uint8')


parser = argparse.ArgumentParser()
parser.add_argument('ntu_path', help='path to NTU RGB+D skeleton zip file')
parser.add_argument('dest', help='path for HDF5 output file')

if __name__ == '__main__':
    args = parser.parse_args()
    with ZipFile(args.ntu_path) as in_fp, h5py.File(args.dest, 'w') as out_fp:
        all_bone_lengths = []

        tq_iter = tqdm(in_fp.namelist(), smoothing=0.005, postfix={'drop': 0})
        dropped = 0
        for seq_path in tq_iter:
            if not seq_path.endswith('.skeleton'):
                continue
            skelly_meta, skelly_tracks, discarded \
                = read_skeleton_file(in_fp, seq_path)
            if discarded:
                dropped += discarded
                tq_iter.set_postfix(drop=dropped)
            ident = skelly_meta['ident']
            if ident in BAD_IDENTIFIERS:
                continue
            perf_id = skelly_meta['performer']
            assert isinstance(perf_id, int) and 1 <= perf_id <= 40
            is_train = perf_id not in EVAL_PERFORMERS
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
                # Confusing terminology, but "colour_x" = "x coordinate of
                # joint in RGB (colour) image"
                skeleton_xy = np.stack(
                    [track.skeletons.colour_x, track.skeletons.colour_y],
                    axis=1)
                scaled_skels, scale = rescale(skeleton_xy)
                num_frames = len(skeleton_xy)
                action_vec = np.full((num_frames, ), action_id).astype('uint8')
                out_fp[prefix + 'poses'] = scaled_skels.astype('float32')
                out_fp[prefix + 'actions'] = action_vec
                out_fp[prefix + 'valid'] = np.ones_like(skeleton_xy) \
                                             .astype('uint8')
                out_fp[prefix + 'scale'] = scale
                out_fp[prefix + 'is_train'] = is_train

                # same, but for 3D data
                prefix_3d = '/seqs3d/' + track_name + '/'
                skeleton_xyz = np.stack(
                    [track.skeletons.x, track.skeletons.y, track.skeletons.z],
                    axis=-1)
                expmap = xyz_to_expmap(skeleton_xyz, JOINT_PARENT_INDS)
                out_fp[prefix_3d + 'skeletons'] = expmap.astype('float32')
                out_fp[prefix_3d + 'actions'] = action_vec
                out_fp[prefix_3d + 'is_train'] = is_train

                # average this out later for display
                all_bone_lengths.append(
                    bone_lengths(skeleton_xyz, JOINT_PARENT_INDS))

        action_names = [n for i, n in sorted(ACTION_CLASSES.items())]
        # use CPM parents
        out_fp['/parents'] = np.array([0, 0, 1, 2, 3, 1, 5, 6], dtype=int)
        out_fp['/action_names'] = h5_json_encode(action_names)
        out_fp['/num_actions'] = len(action_names)

        out_fp['/joint_names_3d'] = h5_json_encode(JOINT_NAMES)
        out_fp['/parents_3d'] = np.array(JOINT_PARENT_INDS, dtype='uint8')
        # store good length for each bone just so that we can visualise motion
        # (expmap parameterisation throws away bone length)
        all_bone_lengths = np.concatenate(all_bone_lengths, axis=0)
        const_bone_lengths = np.median(all_bone_lengths, axis=0)
        assert const_bone_lengths.shape == (len(JOINT_PARENT_INDS), )
        out_fp['/bone_lengths_3d'] = const_bone_lengths.astype('float32')
