"""Generate legacy (develop-format) humanoid fixtures with develop's own code.

Not a test: tests/test_dataio_legacy_convert.py uses its output.  It imports
develop's flat modules (anim.*, util.*, tools.*), so run it from a develop
checkout (torch 1.13 venv), with this repository's tests/fixtures as output:
    cd <develop checkout>
    PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python <this file> <port repo>/tests/fixtures

Writes:
  dataio_legacy_humanoid.pkl      motion file laid out like the output of
                                  tools/motion_file_proc/robot_motion_preprocess.py
                                  (frames, fps, loop_mode, contacts, opt:body_constraints,
                                  hf_mask_inds, terrain)
  dataio_legacy_terrain_only.pkl  terrain-only file like parc_2_kin_gen.py's only_gen
                                  output ({"terrain", "path_nodes"})
  dataio_legacy_humanoid_ref.npz  reference values read back with develop's MotionLib

Tensors computed on the GPU stay GPU tensors in the pickles, as with
robot_motion_preprocess.py --device cuda and parc_2 (only_gen); the port has to
load them on CPU.
"""
import os
import pickle
import sys

import numpy as np
import torch

DEV_ROOT = os.getcwd()
sys.path.insert(0, DEV_ROOT)

import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import util.torch_util as torch_util
from tools.motion_file_proc import robot_motion_preprocess as rmp

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)

char_file = os.path.join(DEV_ROOT, "data/assets/humanoid.xml")
char_model = kin_char_model.KinCharModel(device)
char_model.load_char_file(char_file)

# --- synthetic motion -------------------------------------------------------
num_frames = 20
fps = 30
dof_size = char_model.get_dof_size()
t = np.arange(num_frames, dtype=np.float64) / fps
rng = np.random.RandomState(0)

root_pos = np.stack([0.9 * t,
                     0.05 * np.sin(2.0 * np.pi * t),
                     0.92 + 0.02 * np.cos(4.0 * np.pi * t)], axis=-1)
root_rot = np.stack([0.05 * np.sin(2.0 * np.pi * t),
                     0.03 * np.cos(2.0 * np.pi * t),
                     0.3 + 0.4 * t], axis=-1)  # exp map
amp = rng.uniform(0.1, 0.6, size=dof_size)
freq = rng.uniform(0.5, 2.0, size=dof_size)
phase = rng.uniform(0.0, 2.0 * np.pi, size=dof_size)
joint_dof = amp * np.sin(2.0 * np.pi * freq * t[:, None] + phase)
frames = np.concatenate([root_pos, root_rot, joint_dof], axis=-1).astype(np.float32)

num_bodies = char_model.get_num_joints()
contacts = np.zeros((num_frames, num_bodies), dtype=np.float32)
contacts[0:8, char_model.get_body_id("left_foot")] = 1.0
contacts[13:20, char_model.get_body_id("left_foot")] = 1.0
contacts[5:15, char_model.get_body_id("right_foot")] = 1.0
contacts[9:12, char_model.get_body_id("right_hand")] = 1.0

# --- terrain: a box and a step ------------------------------------------------
terrain = terrain_util.SubTerrain("terrain", x_dim=24, y_dim=12, dx=0.1, dy=0.1,
                                  min_x=-0.5, min_y=-0.6, device=device)
terrain.hf[14:, :] = 0.3
terrain.hf[8:10, 3:6] = 0.15

# --- derived data, computed the way robot_motion_preprocess.py does ------------
frames_t = torch.tensor(frames, dtype=torch.float32, device=device)
contacts_t = torch.tensor(contacts, dtype=torch.float32, device=device)
root_pos_t = frames_t[..., 0:3]
root_rot_t = torch_util.exp_map_to_quat(frames_t[..., 3:6])
joint_rot_t = char_model.dof_to_rot(frames_t[..., 6:])

body_constraints = rmp.compute_body_constraints_generic(
    root_pos=root_pos_t, root_rot=root_rot_t, joint_rot=joint_rot_t, contacts=contacts_t,
    char_model=char_model, terrain=terrain, robot_type=None)

body_points = geom_util.get_char_point_samples(
    char_model, sphere_num_subdivisions=0, box_num_slices=2, box_dim_x=2, box_dim_y=2,
    capsule_num_circle_points=4, capsule_num_sphere_subdivisons=0, capsule_num_cylinder_slices=2)
hf_mask_inds = terrain_util.compute_hf_extra_vals(
    motion_frames=frames_t, terrain=terrain, char_model=char_model, char_body_points=body_points)

processed_data = {
    "frames": frames_t.cpu().numpy(),
    "fps": fps,
    "loop_mode": "CLAMP",
    "contacts": contacts_t.cpu().numpy(),
    "opt:body_constraints": body_constraints,
    "hf_mask_inds": hf_mask_inds,
    "terrain": terrain,
}
terrain.to_torch("cpu")  # as robot_motion_preprocess.py does before saving
motion_path = os.path.join(out_dir, "dataio_legacy_humanoid.pkl")
with open(motion_path, "wb") as f:
    pickle.dump(processed_data, f)

# --- terrain-only file, as parc_2_kin_gen.py writes with only_gen ---------------
only_terrain = terrain_util.SubTerrain("terrain", x_dim=6, y_dim=5, dx=0.2, dy=0.2,
                                       min_x=-0.4, min_y=-0.5, device=device)
only_terrain.hf[3:, 1:4] = 0.45
path_nodes = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.1, 0.45], [0.6, 0.2, 0.45]],
                          dtype=torch.float32, device=device)
with open(os.path.join(out_dir, "dataio_legacy_terrain_only.pkl"), "wb") as f:
    pickle.dump({"terrain": only_terrain, "path_nodes": path_nodes}, f)

# --- reference values from develop's MotionLib ---------------------------------
ref_char_model = kin_char_model.KinCharModel("cpu")
ref_char_model.load_char_file(char_file)
mlib = motion_lib.MotionLib(motion_path, ref_char_model, "cpu", contact_info=True)
ref_terrain = mlib._terrains[0]
mask_inds = [x.cpu().numpy().astype(np.int64) for x in mlib._hf_mask_inds[0]]

constraint_rows = []
for body_id, constraints in enumerate(processed_data["opt:body_constraints"]):
    for c in constraints:
        point = c.constraint_point.detach().cpu().numpy().astype(np.float64)
        constraint_rows.append([body_id, c.start_frame_idx, c.end_frame_idx] + point.tolist())
assert len(constraint_rows) > 0, "fixture needs at least one body constraint"

np.savez_compressed(
    os.path.join(out_dir, "dataio_legacy_humanoid_ref.npz"),
    num_frames=mlib._motion_num_frames.cpu().numpy(),
    fps=mlib._motion_fps.cpu().numpy(),
    loop_modes=mlib._motion_loop_modes.cpu().numpy(),
    root_pos=mlib._frame_root_pos.cpu().numpy(),
    root_rot=mlib._frame_root_rot.cpu().numpy(),
    joint_rot=mlib._frame_joint_rot.cpu().numpy(),
    dof=mlib.joint_rot_to_dof(mlib._frame_joint_rot).cpu().numpy(),
    frames=mlib._motion_frames.cpu().numpy(),
    contacts=mlib._frame_contacts.cpu().numpy(),
    hf=ref_terrain.hf.cpu().numpy(),
    hf_maxmin=ref_terrain.hf_maxmin.cpu().numpy(),
    min_point=ref_terrain.min_point.cpu().numpy(),
    dxdy=ref_terrain.dxdy.cpu().numpy(),
    hf_mask_inds=np.concatenate(mask_inds, axis=0),
    hf_mask_inds_lengths=np.array([x.shape[0] for x in mask_inds], dtype=np.int64),
    body_constraints=np.array(constraint_rows, dtype=np.float64),
)
print("num frames", num_frames, "dof", dof_size, "bodies", num_bodies,
      "constraints", len(constraint_rows), "mask inds", sum(x.shape[0] for x in mask_inds))
for name in sorted(os.listdir(out_dir)):
    print(name, os.path.getsize(os.path.join(out_dir, name)))
