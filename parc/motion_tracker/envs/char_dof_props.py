"""Per-DOF joint limits and PD gains for the simulated character, computed from the kinematic model.

Arrays are in the kinematic model's DOF order, which is also IsaacGym's (IGCharEnv checks this).
This module does not import IsaacGym, so it can be tested on the CPU.
"""
from typing import List, Mapping, Optional, Tuple

import numpy as np

from parc.util.logger import Logger

PD_PARAM_KEYS = ("stiffness", "damping")


def get_dof_joint_names(char_model) -> List[str]:
    """Name of the joint that owns each DOF."""
    names = []
    for j in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(j)
        names += [joint.name] * joint.get_dof_dim()
    assert len(names) == char_model.get_dof_size()
    return names


def get_kin_dof_limits(char_model) -> Tuple[np.ndarray, np.ndarray]:
    """Lower and upper limit of each DOF in radians, as float32 arrays of length dof_size."""
    dof_size = char_model.get_dof_size()
    lower = np.full(dof_size, np.nan, dtype=np.float32)
    upper = np.full(dof_size, np.nan, dtype=np.float32)
    for j in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(j)
        dof_dim = joint.get_dof_dim()
        if (dof_dim == 0):
            continue
        assert joint.limits is not None, "Joint {} has no limits".format(joint.name)
        # hinge limits are [2], spherical limits [3, 2]
        limits = joint.limits.detach().cpu().numpy().reshape(dof_dim, 2)
        lower[joint.dof_idx:joint.dof_idx + dof_dim] = limits[:, 0]
        upper[joint.dof_idx:joint.dof_idx + dof_dim] = limits[:, 1]

    assert not (np.any(np.isnan(lower)) or np.any(np.isnan(upper))), "Some DOFs have no joint limits"
    return lower, upper


def match_pd_params(joint_name: str, params: Mapping) -> List[str]:
    """Keys of params, in config order, that are case-insensitive substrings of joint_name.
    The first one sets the joint's gain."""
    name = joint_name.lower()
    return [key for key in params if str(key).lower() in name]


def build_pd_gains(char_model, default_kp, default_kd, pd_params: Optional[Mapping]):
    """Per-DOF PD gains: default_kp/default_kd (the asset's values), overridden per joint from pd_params.

    pd_params is the env config's {"stiffness": {name: kp}, "damping": {name: kd}}. Each joint takes the
    first key that is a case-insensitive substring of its name (see match_pd_params), for all of its DOFs.
    scripts/parc_3_tracker.py and parc_4_phys_record.py re-save the env config with sorted keys, so a
    joint that matches several keys with different values gets a warning.

    Returns:
        kp, kd: float32 arrays of length dof_size.
        gain_keys: for each DOF, the (stiffness key, damping key) it took, None where the default was kept.
    """
    dof_size = char_model.get_dof_size()
    kp = np.array(default_kp, dtype=np.float32).reshape(-1)
    kd = np.array(default_kd, dtype=np.float32).reshape(-1)
    assert kp.shape[0] == dof_size and kd.shape[0] == dof_size, \
        "Expected {} default gains, got {} kp and {} kd".format(dof_size, kp.shape[0], kd.shape[0])
    gain_keys = [(None, None)] * dof_size

    if (not pd_params):
        return kp, kd, gain_keys

    unknown_keys = [key for key in pd_params if key not in PD_PARAM_KEYS]
    if (len(unknown_keys) > 0):
        raise ValueError("Unknown pd_params keys {}, expected {}".format(unknown_keys, list(PD_PARAM_KEYS)))
    groups = [(group, pd_params.get(group, None) or {}) for group in PD_PARAM_KEYS]
    for group, params in groups:
        for key, value in params.items():
            if (float(value) < 0.0):
                raise ValueError("pd_params.{}.{} must be >= 0, got {}".format(group, key, value))

    used_keys = set()
    for j in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(j)
        dof_dim = joint.get_dof_dim()
        if (dof_dim == 0):
            continue
        dofs = slice(joint.dof_idx, joint.dof_idx + dof_dim)
        joint_keys = []
        for (group, params), gains in zip(groups, (kp, kd)):
            keys = match_pd_params(joint.name, params)
            if (len(set(float(params[key]) for key in keys)) > 1):
                Logger.print("Warning: joint {} matches pd_params.{} keys {}; using {}".format(
                    joint.name, group, keys, keys[0]))
            if (len(keys) > 0):
                gains[dofs] = float(params[keys[0]])
                used_keys.add((group, keys[0]))
            joint_keys.append(keys[0] if len(keys) > 0 else None)
        gain_keys[dofs] = [tuple(joint_keys)] * dof_dim

    unused_keys = ["{}.{}".format(group, key) for group, params in groups
                   for key in params if (group, key) not in used_keys]
    if (len(unused_keys) > 0):
        Logger.print("Warning: pd_params keys that set no joint: {}".format(", ".join(unused_keys)))
    return kp, kd, gain_keys
