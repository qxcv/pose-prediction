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
    paper, etc. Stores inter-frame offset in root joint position."""
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
            parent_bones = xyz_seq[:, grandparent] - xyz_seq[:, parent]

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


def exp_to_rotmat(exp):
    """Convert rotation paramterised as exponential map into ordinary 3x3
    rotation matrix."""
    assert exp.shape == (3, ), "was expecting expmap vector"

    # begin by normalising all exps
    angle = np.linalg.norm(exp)
    if angle < 1e-5:
        # assume no rotation
        return np.eye(3)
    dir = exp / angle

    # Rodrigues' formula, matrix edition
    K = np.array([[0, -dir[2], dir[1]], [dir[2], 0, -dir[0]],
                  [-dir[1], dir[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


def expmap_to_xyz(exp_seq, parents, bone_lengths):
    """Inverse of xyz_to_expmap. Won't be able to recover initial offset."""
    assert exp_seq.ndim >= 2 and exp_seq.shape[-1] == 3, \
        "Wanted TxJx3 array containing T expmap skeletons, each with J dirs."

    toposorted = toposort(parents)
    root = toposorted[0]
    xyz_seq = np.zeros_like(exp_seq)

    # restore head first
    xyz_seq[:, root, :] = np.cumsum(exp_seq[:, root, :], axis=1)

    # simultaneously recover bones (normalised offset from parent) and original
    # coordinates
    bones = np.zeros_like(exp_seq)
    bones[:, root, 1] = -1
    for child in toposorted[1:]:
        parent = parents[child]
        parent_bone = bones[:, parent, :]
        exps = exp_seq[:, child, :]
        for t in range(len(exp_seq)):
            # might be able to vectorise this, but may take too long to justify
            R = exp_to_rotmat(exps[t])
            bones[t, child] = np.dot(R, parent_bone[t])
        scaled_child_bones = bones[:, child] * bone_lengths[child]
        xyz_seq[:, child] = xyz_seq[:, parent] + scaled_child_bones

    return xyz_seq


def exps_to_quats(exps):
    """Turn tensor of exponential map angles into quaternions. If using with
    {xyz,expmap}_to_{expmap,xyz}, remember to remove root node before using
    this!"""
    # See
    # https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula#Rotation_angle_and_rotation_axis

    # flatten the matrix to save my own sanity (numpy Boolean array indexing is
    # super confusing)
    num_exps = int(np.prod(exps.shape[:-1]))
    assert exps.shape[-1] == 3
    exps_flat = exps.reshape((num_exps, 3))
    rv_flat = np.zeros((num_exps, 4))

    # get angles & set zero-rotation vecs to be zero-rotation quaternions (w=1)
    angles = np.linalg.norm(exps_flat, axis=-1)
    zero_mask = angles < 1e-5
    rv_flat[zero_mask, 0] = 1

    # everthing else gets a meaningful value
    nonzero_mask = ~zero_mask
    nonzero_angles = angles[nonzero_mask]
    nonzero_exps_flat = exps_flat[nonzero_mask, :]
    nonzero_normed = nonzero_exps_flat / nonzero_angles[..., None]
    sines = np.sin(nonzero_angles / 2)
    rv_flat[nonzero_mask, 0] = np.cos(nonzero_angles / 2)
    rv_flat[nonzero_mask, 1:] = nonzero_normed * sines[..., None]

    rv_shape = exps.shape[:-1] + (4, )
    return rv_flat.reshape(rv_shape)


def quats_to_eulers(quats):
    """Turn tensor of quaternions into Euler angles."""
    # See
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion
    # This formula is ZYX/yaw-pitch-roll (which I understand to be a Tait-Bryan
    # layout?)
    rv_shape = quats.shape[:-1] + (3, )
    rv = np.zeros(rv_shape)
    # split out for convenience
    q0, q1, q2, q3 = [quats[..., i] for i in range(4)]
    rv[..., 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rv[..., 1] = np.arcsin(2 * (q0 * q2 - q1 * q3))
    rv[..., 2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rv


def exps_to_eulers(exps):
    return quats_to_eulers(exps_to_quats(exps))


def test_exps_to_eulers():
    # http://www.andre-gaschler.com/rotationconverter/ is awesome! I used it to
    # come up with these test cases.
    v1 = np.full((3, ), np.sqrt(3) / 3)
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    v4 = np.array([0, 0, -1])
    test_cases = [(0.2 * v1, (0.1223661, 0.1082687, 0.1223661)),
                  (-0.95 * v1, (-0.4293868, -0.6548807, -0.4293868)),
                  (0.2 * v2, (0.2, 0, 0)), (0.3 * v3,
                                            (0, 0.3, 0)), (0.4 * v4,
                                                           (0, 0, -0.4))]
    for exp, target in test_cases:
        # test individually
        converted = exps_to_eulers(exp)
        assert np.allclose(converted, target)

    # also test as one big matrix (with some useless dimensions) to make sure
    # dimension handling is good
    exp_mat = np.stack([e for e, t in test_cases], axis=0)[None, :, None, :]
    target_mat = np.stack([t for e, t in test_cases], axis=0)[None, :, None, :]
    converted_mat = exps_to_eulers(exp_mat)
    assert np.allclose(converted_mat, target_mat)


def plot_xyz_skeleton(skeleton_xyz, parents, mp3d_axes, handles=None):
    """Plot an xyz-parameterised skeleton using some given Matplotlib 3D
    axes. Will return old handles, which can be used to """
    assert skeleton_xyz.shape == (len(parents), 3), "need J*3 coords matrix"
    toposorted = toposort(parents)

    if handles is None:
        handles = []
        mp3d_axes.set_aspect('equal')
        for child in toposorted[1:]:
            parent = parents[child]
            coords = skeleton_xyz[[parent, child], :]
            h, = mp3d_axes.plot(
                coords[:, 0], coords[:, 1], zs=coords[:, 2], zdir='y')
            handles.append(h)

        mp3d_axes.set_xlabel('x')
        mp3d_axes.set_ylabel('y')
        mp3d_axes.set_zlabel('z')

        # ridiculous trick from http://stackoverflow.com/a/21765085
        # draws a box around the data so that matplot3d uses equal aspect ratio
        x_max, x_min = skeleton_xyz[:, 0].max(), skeleton_xyz[:, 0].min()
        y_max, y_min = skeleton_xyz[:, 1].max(), skeleton_xyz[:, 1].min()
        z_max, z_min = skeleton_xyz[:, 2].max(), skeleton_xyz[:, 2].min()
        # swap z/y
        max_range = max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.0

        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        mp3d_axes.set_xlim(mid_x - max_range, mid_x + max_range)
        # swap y/z because zdir='y'
        mp3d_axes.set_zlim(mid_y - max_range, mid_y + max_range)
        mp3d_axes.set_ylim(mid_z - max_range, mid_z + max_range)

        # look at person from above
        mp3d_axes.elev = 30
        mp3d_axes.azim = 60

        mp3d_axes.invert_zaxis()
    else:
        for handle, child in zip(handles, toposorted[1:]):
            parent = parents[child]
            coords = skeleton_xyz[[parent, child], :]
            handle.set_xdata(coords[:, 0])
            handle.set_ydata(coords[:, 1])
            # there's no set_zdata for some reason :/
            handle.set_3d_properties(zs=coords[:, 2], zdir='y')

    return handles
