"""Writes charmodel_g1_develop_reference.npz with develop's KinCharModel (pre-migration layout).

Run from a develop checkout with develop's environment (torch 1.13), e.g.:
    cd <develop checkout> && PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python \
        <repo>/tests/fixtures/charmodel_g1_develop_reference.py --develop-root . --out <repo>/tests/fixtures/charmodel_g1_develop_reference.npz
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--develop-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    develop_root = os.path.abspath(args.develop_root)
    sys.path.insert(0, develop_root)
    import anim.kin_char_model as kin_char_model  # develop layout
    import util.geom_util as geom_util
    import util.torch_util as torch_util

    char_file = os.path.join(develop_root, "data", "assets", "unitree_g1", "g1_mocap_29dof.xml")
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(char_file)

    lower = char_model._lower_dof_limits
    upper = char_model._upper_dof_limits
    gen = torch.Generator().manual_seed(20261014)
    num_poses = 32
    root_pos = torch.rand(num_poses, 3, generator=gen) * 2.0 - 1.0
    root_rot = torch_util.quat_normalize(torch.randn(num_poses, 4, generator=gen))
    dof = lower + (upper - lower) * torch.rand(num_poses, lower.shape[0], generator=gen)
    dof[0] = 0.0

    joint_rot = char_model.dof_to_rot(dof)
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    points = geom_util.get_minimal_char_point_samples(char_model)

    joints = char_model._joints
    np.savez_compressed(
        args.out,
        torch_version=np.array(torch.__version__),
        body_names=np.array(char_model.get_body_names()),
        counts=np.array([len(char_model.get_body_names()), char_model.get_num_joints(),
                         char_model.get_dof_size()], dtype=np.int64),
        joint_names=np.array([j.name for j in joints]),
        joint_types=np.array([j.joint_type.value for j in joints], dtype=np.int64),
        joint_dof_idx=np.array([j.dof_idx for j in joints], dtype=np.int64),
        lower_dof_limits=lower.numpy(),
        upper_dof_limits=upper.numpy(),
        in_root_pos=root_pos.numpy(),
        in_root_rot=root_rot.numpy(),
        in_dof=dof.numpy(),
        joint_rot=joint_rot.numpy(),
        fk_body_pos=body_pos.numpy(),
        fk_body_rot=body_rot.numpy(),
        min_points_counts=np.array([p.shape[0] for p in points], dtype=np.int64),
        min_points=torch.cat(points, dim=0).numpy(),
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
