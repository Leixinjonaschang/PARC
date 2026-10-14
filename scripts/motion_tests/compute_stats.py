import sys

sys.path.insert(1, sys.path[0] + ("/../.."))

import pickle

import torch
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib

#from diffusion.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler

motion_lib_file = "../Data/parkour_dataset/pre_auto_aug_dataset.yaml"
char_file = "data/assets/humanoid.xml"
device = "cpu"
char_model = kin_char_model.KinCharModel(device)
char_model.load_char_file(char_file)
mlib = motion_lib.MotionLib(motion_lib_file, char_model, device, contact_info=True)



key_body_names = ["left_foot", "right_foot", "left_hand", "right_hand"]
key_body_ids = []
for name in key_body_names:
    key_body_ids.append(char_model.get_body_id(name))
key_body_ids = torch.tensor(key_body_ids, dtype=torch.int64, device=device)

dt = 1.0 / 30.0

all_body_vels = []
all_body_accs = []
all_body_jerks = []

# compute velocity, acceleration, and jerk for every frame
for id in mlib._motion_ids:
    #print("ID:", id.item())

    # TODO fix
    root_pos, root_rot, joint_rot, body_pos, body_rot = mlib.get_frames_for_id(id)


    body_vels = (body_pos[1:] - body_pos[:-1]) / dt
    body_accs = (body_vels[1:] - body_vels[:-1]) / dt
    body_jerks = (body_accs[1:] - body_accs[:-1]) / dt

    all_body_vels.append(body_vels)
    all_body_accs.append(body_accs)
    all_body_jerks.append(body_jerks)

all_body_vels = torch.cat(all_body_vels, dim=0)
all_body_accs = torch.cat(all_body_accs, dim=0)
all_body_jerks = torch.cat(all_body_jerks, dim=0)


vel_mag = all_body_vels.norm(dim=-1)
acc_mag = all_body_accs.norm(dim=-1)
jerk_mag = all_body_jerks.norm(dim=-1)

print("Max speed:", torch.max(vel_mag))
print("Max acceleration (magnitude):", torch.max(acc_mag))
print("Max jerk (magnitude):", torch.max(jerk_mag))

print("Mean speed:", torch.mean(vel_mag))
print("Mean acceleration (magnitude):", torch.mean(acc_mag))
print("Mean jerk (magnitude):", torch.mean(jerk_mag))

print("Std speed:", torch.std(vel_mag))
print("Std acceleration (magnitude):", torch.std(acc_mag))
print("Std jerk (magnitude):", torch.std(jerk_mag))

# save stats to a pkl file

output = {
    "vel": all_body_vels.cpu().numpy(),
    "acc": all_body_accs.cpu().numpy(),
    "jerk": all_body_jerks.cpu().numpy()
}

with open("output/body_pos_derivatives.txt", "wb") as f:
    pickle.dump(output, f)


quantiles = [0.999, 0.99, 0.95, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
quantiles = torch.tensor(quantiles, dtype=torch.float32, device=device)
jerk_quantiles = torch.quantile(jerk_mag, quantiles)
acc_quantiles = torch.quantile(acc_mag, quantiles)
vel_quantiles = torch.quantile(vel_mag, quantiles)

for i in range(quantiles.shape[0]):
    print("Quantile:", quantiles[i].item() * 100.0)
    print("Jerk:", jerk_quantiles[i].item())
    print("Acc:", acc_quantiles[i].item())
    print("Vel:", vel_quantiles[i].item())
    print("")