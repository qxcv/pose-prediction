"""Code for converting (x, y, z) skeleton parameterisations into exponential
map parameterisations."""

import numpy as np


def _toposort_visit(parents, visited, toposorted, joint):
    parent = parents[joint]
    visited[joint] = True
    if parent != joint and not visited[parent]:
        _toposort_visit(parents, visited, toposorted, parent)
    toposorted.append(joint)


def check_toposorted(parents, toposorted):
    # check that array contains all/only joint indices
    assert sorted(toposorted) == list(range(len(parents)))

    # make sure that order is correct
    to_topo_order = {
        joint: topo_order
        for topo_order, joint in enumerate(toposorted)
    }
    for joint in toposorted:
        assert to_topo_order[joint] >= to_topo_order[parents[joint]]

    # verify that we have only one root
    joints = range(len(parents))
    assert sum(parents[joint] == joint for joint in joints) == 1


def toposort(parents):
    """Return toposorted array of joint indices (sorted root-first)."""
    toposorted = []
    visited = np.zeros_like(parents, dtype=bool)
    for joint in range(len(parents)):
        if not visited[joint]:
            _toposort_visit(parents, visited, toposorted, joint)

    check_toposorted(parents, toposorted)

    return np.asarray(toposorted)


def bone_lengths(xyz_skels, parents):
    # takes ...xJx3 array of (x, y, z) skeletons and returns. Returns ...xJ
    # array of lengths, where the jth entry gives the length of the bone with
    # joint j at the far end (hence, entry for root is always zero). "..."
    # indicates that the operation is broadcast over leading dimensions in the
    # way you'd expect.
    assert xyz_skels.shape[-2:] == (len(parents), 3), \
        "expect last two dims to give joints and xyz pos"

    lengths = np.zeros(xyz_skels.shape[:-1])
    for child in toposort(parents)[1:]:
        parent = parents[child]
        child_locs = xyz_skels[..., child, :]
        parent_locs = xyz_skels[..., parent, :]
        diffs = parent_locs - child_locs
        cp_lengths = np.linalg.norm(diffs, axis=-1)
        lengths[..., child] = cp_lengths

    return lengths


def _norm_bvecs(bvecs):
    """Norm bone vectors, handling small magnitudes by zeroing bones."""
    bnorms = np.linalg.norm(bvecs, axis=-1)
    mask_out = bnorms <= 1e-5
    # implicit broadcasting is deprecated (?), so I'm doing this instead
    _, broad_mask = np.broadcast_arrays(bvecs, mask_out[..., None])
    bvecs[broad_mask] = 0
    bnorms[mask_out] = 1
    return bvecs / bnorms[..., None]


def xyz_to_expmap(xyz_seq, parents):
    """Converts a tree of (x, y, z) positions into the parameterisation used in
    the SRNN paper, "modelling human motion with binary latent variables"
    paper, etc. Chops off the first coordinate (not needed)."""
    assert xyz_seq.ndim == 3 and xyz_seq.shape[2] == 3, \
        "Wanted TxJx3 array containing T skeletons, each with J (x, y, z)s"

    exp_seq = np.zeros_like(xyz_seq)
    toposorted = toposort(parents)
    # [1:] ignores the root; apart from that, processing order doesn't actually
    # matter
    for child in toposorted[1:]:
        parent = parents[child]
        bones = xyz_seq[:, parent] - xyz_seq[:, child]
        grandparent = parents[parent]
        if grandparent == parent:
            # we're the root; parent bones will be constant (x,y,z)=(0,-1,0)
            parent_bones = np.zeros_like(bones)
            parent_bones[:, 1] = -1
        else:
            # we actually have a parent bone :)
            parent_bones = xyz_seq[:, parent] - xyz_seq[:, child]

        # normalise parent and child bones
        norm_bones = _norm_bvecs(bones)
        norm_parent_bones = _norm_bvecs(parent_bones)
        # cross product will only be used to get axis around which to rotate
        cross_vecs = np.cross(norm_parent_bones, norm_bones)
        norm_cross_vecs = _norm_bvecs(cross_vecs)
        # dot products give us rotation angle
        angles = np.arccos(np.sum(norm_bones * norm_parent_bones, axis=-1))
        log_map = norm_cross_vecs * angles[..., None]
        exp_seq[:, child] = log_map

    # root will store distance from previous frame
    root = toposorted[0]
    exp_seq[1:, root] = xyz_seq[1:, root] - xyz_seq[:-1, root]

    return exp_seq


def expmap_to_xyz(exp_seq, parents):
    """Inverse of xyz_to_expmap. Won't be able to recover initial offset."""
    assert exp_seq.ndim == 3 and exp_seq.shape[2] == 3, \
        "Wanted TxJx3 array containing T expmap skeletons, each with J dirs."

    toposorted = toposort(parents)
    root = toposorted[0]
    xyz_seq = np.zeros_like(exp_seq)

    # restore head first
    xyz_seq[:, root, :] = np.cumsum(exp_seq[:, root, :], axis=1)

    raise NotImplementedError('Need to finish this')

    return xyz_seq
