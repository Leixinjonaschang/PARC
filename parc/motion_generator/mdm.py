import copy
import enum
import os
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.motion_generator.diffusion_util as diffusion_util
import parc.motion_generator.utils.EMA as EMA
import parc.motion_generator.utils.rot_changer as rot_changer
import parc.util.geom_util as geom_util
import parc.util.path_loader as path_loader
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from parc.motion_generator.diffusion_util import (
    MDMFrameType,
    MDMKeyType,
    RelativeZStyle,
    TargetInfo,
)
from parc.motion_generator.mdm_transformer import MDMTransformer
from parc.motion_generator.base_generator import MotionGenerator
from parc.motion_generator.motion_sampler import MotionSampler
from parc.util.logger import Logger

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# references:
# https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
# https://huggingface.co/blog/annotated-diffusion

class PredictMode(enum.Enum):
    PREDICT_X0 = 0
    PREDICT_NOISE = 1

class LossType(enum.Enum):
    SIMPLE_ROOT_POS_LOSS = 0
    SIMPLE_ROOT_ROT_LOSS = 1
    SIMPLE_JOINT_ROT_LOSS = 2
    SIMPLE_CONTACT_LOSS = 3
    VEL_ROOT_POS_LOSS = 4
    VEL_ROOT_ROT_LOSS = 5
    VEL_JOINT_ROT_LOSS = 6
    FK_BODY_POS_LOSS = 7
    FK_BODY_ROT_LOSS = 8
    FOOT_CONTACT_LOSS = 9
    TARGET_LOSS = 10
    HF_COLLISION_LOSS = 11
    SIMPLE_FLOOR_HEIGHT_LOSS = 12
    SIMPLE_BODY_POS_LOSS = 13
    BODY_POS_CONSISTENCY_LOSS = 14
    VEL_FK_BODY_POS_LOSS = 15

class GenerationMode(enum.Enum):
    MODE_REVERSE_DIFFUSION = 0
    MODE_DDIM = 1
    MODE_ECM = 2
    NONE = 3

class TargetType(enum.Enum):
    XY_POS = 0
    XY_POS_AND_HEADING = 1
    XY_POS_AND_DELTA_FLOOR_Z = 2
    XY_DIR = 3

def get_dir_from_motion(motions, eps=0.05):
    dir = motions[:, -1:, 0:2]
    dir_length = torch.norm(dir, dim=-1, keepdim=True)
    standing_still = dir_length < eps
    target_dir = torch.where(standing_still, torch.zeros_like(dir), dir/dir_length)

    return target_dir

def get_dir_from_canonicalized_pos_and_rot(pos, rot, pos_eps=0.1, heading_eps=0.25):
    # more description:
    # pos and rotation are unnormalized and already canonicalized
    pos = pos[..., 0:2]
    dir_length = torch.norm(pos, dim=-1, keepdim=True)
    standing_still_check1 = dir_length < pos_eps

    heading = torch_util.calc_heading(rot)
    heading_unsqueezed = heading.unsqueeze(-1)
    standing_still_check2 = heading_unsqueezed < heading_eps

    divide_by_zero_check = dir_length < 0.0001

    standing_still = torch.logical_and(standing_still_check1, standing_still_check2)

    standing_still = torch.logical_or(standing_still, divide_by_zero_check)

    target_dir = torch.where(standing_still, torch.zeros_like(pos), pos/dir_length)

    use_heading_dir = torch.logical_and(standing_still_check1, torch.logical_not(standing_still_check2))
    heading_dir = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    target_dir = torch.where(use_heading_dir, heading_dir, target_dir)

    return target_dir

def pseudo_huber_loss_fn(x, y):
        # Pseudo-Huber metric loss function
        c = 0.03
        c2 = 0.0009
        loss = (x - y) ** 2 + c2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        return torch.sqrt(loss) - c # sqrt so outliers are not as impactful?
    
def squared_l2_loss_fn(x, y):
    loss = (x - y)**2
    loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
    return loss

class MDM(MotionGenerator):
    def __init__(self, cfg):
        # Motion Sampler
        self._cfg = cfg # copy config here for when we wanna copy this model
        self._use_wandb = cfg['use_wandb']
        self._device = cfg['device']
        self._kin_char_model = kin_char_model.KinCharModel(self._device)
        # TODO: save kin_char_model xml with the checkpoint as a string
        self._kin_char_model.load_char_file(cfg['char_file'])
        self._diffusion_timesteps = cfg['diffusion_timesteps']
        self._epochs = cfg['epochs']
        self._epochs_per_checkpoint = cfg['epochs_per_checkpoint']
        self._batch_size = cfg['batch_size']
        self._dropout = cfg['dropout']
        self._iters_per_epoch = cfg['iters_per_epoch']
        self._test_size = cfg['test_size']
        self._lr = cfg['lr'] #for transformer #5e-5#for MLP
        self._weight_decay = cfg['weight_decay']
        self._batch_size = cfg['batch_size']
        self._obs_dropout = cfg['obs_dropout']
        self._target_dropout = cfg['target_dropout']
        self._d_model = cfg['d_model']
        self._num_heads = cfg['num_heads']
        self._d_hid = cfg['d_hid']
        self._num_layers = cfg['num_layers']
        self._sequence_duration = cfg['sequence_duration']
        self._sequence_fps = cfg['sequence_fps']
        self._num_prev_states = cfg['num_prev_states'] # the number of previous states to condition on
        self._prev_state_noise_ind_chance = cfg['prev_state_noise_ind_chance']
        self._prev_state_attention_dropout = cfg['prev_state_attention_dropout'] # different from noise ind, helps prevent overfitting

        self._seq_len = cfg['seq_len']

        # geometric loss weights
        self._w_simple_root_pos = cfg['w_simple_root_pos']
        self._w_simple_root_rot = cfg['w_simple_root_rot']
        self._w_simple_joint_rot = cfg['w_simple_joint_rot']
        self._w_simple_contacts = cfg['w_simple_contacts']
        self._w_vel_root_pos = cfg['w_vel_root_pos']
        self._w_vel_root_rot = cfg['w_vel_root_rot']
        self._w_vel_joint_rot = cfg['w_vel_joint_rot']
        self._w_body_pos = cfg['w_body_pos']
        self._w_body_rot = cfg['w_body_rot']
        self._w_body_vel = cfg['w_body_vel']
        self._w_hf = cfg['w_hf']
        self._w_simple_body_pos = cfg['w_simple_body_pos']
        self._w_body_pos_consistency = cfg['w_body_pos_consistency']
        
        self._target_dir_len_eps = cfg['target_dir_len_eps']
        self._target_dir_heading_eps = cfg['target_dir_heading_eps']

        self._use_simple_loss = True
        self._use_vel_loss = True
        self._use_pos_loss = cfg['use_pos_loss']
        self._use_hf_collision_loss = cfg['use_hf_collision_loss']
        
        self._diffusion_rates = diffusion_util.DiffusionRates(self._diffusion_timesteps, self._device)      
        
        if "custom_num_contacts" in cfg:
            self._custom_num_contacts = cfg["custom_num_contacts"]
            
        features_cfg = cfg["features"]
        rot_type = rot_changer.RotationType[features_cfg["rot_type"]]
        self._rot_changer = rot_changer.RotChanger(rot_type, self._kin_char_model)
        frame_components = features_cfg["frame_components"]
        self._register_mdm_features(frame_components)

        ## LOAD CONDITION OPTIONS ##
        self._use_heightmap_obs = cfg["use_heightmap_obs"]
        grid_dims = cfg["heightmap"]["local_grid"]
        grid_dim_x = 1 + grid_dims["num_x_neg"] + grid_dims["num_x_pos"]
        grid_dim_y = 1 + grid_dims["num_y_neg"] + grid_dims["num_y_pos"]
        self._dx = cfg["heightmap"]["horizontal_scale"]
        self._dy = self._dx
        self._dxdy = torch.tensor([self._dx, self._dy], dtype=torch.float32, device=self._device)
        self._num_x_neg = grid_dims["num_x_neg"]
        self._num_x_pos = grid_dims["num_x_pos"]
        self._num_y_neg = grid_dims["num_y_neg"]
        self._num_y_pos = grid_dims["num_y_pos"]
        self._min_point = torch.tensor([-self._dx * self._num_x_neg, -self._dy * self._num_y_neg], dtype=torch.float32, device=self._device)
        self._grid_dim_x = grid_dim_x
        self._grid_dim_y = grid_dim_y
        self._obs_dim = grid_dim_x * grid_dim_y
        self._max_h = cfg["heightmap"]["max_h"]
        self._min_h = -self._max_h

        self._relative_z_style = RelativeZStyle[cfg["relative_z_style"]]

        self._use_target_obs = cfg["use_target_obs"]
        self._use_target_loss = cfg["use_target_loss"]
        self._w_target = cfg['w_target']
        self._target_type = TargetType[cfg['target_type']]
        if self._target_type == TargetType.XY_POS:
            self._target_dim = 2 # currently just xy position
        elif self._target_type == TargetType.XY_POS_AND_HEADING:
            self._target_dim = 3
        elif self._target_type == TargetType.XY_POS_AND_DELTA_FLOOR_Z:
            self._target_dim = 3
        elif self._target_type == TargetType.XY_DIR:
            self._target_dim = 2
        else:
            assert False

        target_mlp_layers = cfg["target_mlp_layers"]

        in_mlp_layers = cfg["in_mlp_layers"]
        out_mlp_layers = cfg["out_mlp_layers"]

        # options: ddim, ddpm
        self._test_mode = GenerationMode[cfg["test_mode"]]
        self._test_ddim_stride = cfg["test_ddim_stride"]
        
        # build transformer encoder model
        self._denoise_model = MDMTransformer(
            out_dim=self._motion_frame_dof, 
            in_dim=self._motion_frame_dof, 
            d_model=self._d_model, 
            num_heads=self._num_heads, 
            d_hid=self._d_hid, 
            num_layers=self._num_layers, 
            gen_seq_len=self._seq_len,
            diffusion_timesteps=self._diffusion_timesteps,
            dropout=self._dropout,
            num_prev_states=self._num_prev_states,
            use_obs=self._use_heightmap_obs,
            use_target=self._use_target_obs,
            target_dim=self._target_dim,
            target_mlp_layers=target_mlp_layers,
            cnn_cfg = cfg["cnn"],
            in_mlp_layers=in_mlp_layers,
            out_mlp_layers=out_mlp_layers
        )

        self._denoise_model = self._denoise_model.to(self._device)
        self.use_ema = cfg.get("EMA", False)
        if self.use_ema:
            print("Using EMA")
            self.ema_step = 0
            self.ema_decay = cfg['EMA']['ema_decay']
            self.ema_start = cfg['EMA']['ema_start']
            self.ema_update_rate = cfg['EMA']['ema_update_rate']
            self._ema_denoise_model= copy.deepcopy(self._denoise_model)
            self.ema = EMA.EMA(self.ema_decay)

        self._optimizer = torch.optim.AdamW(self._denoise_model.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        # Gets set once train is called
        self._dof_high = None

        self._predict_mode = PredictMode[cfg["predict_mode"]]

        if self._use_hf_collision_loss:
            self.init_char_point_samples()

        self._logger = Logger()
        return

    def update(self):
        if self.use_ema:
            self.update_ema()
        return

    def update_ema(self):
        self.ema_step += 1
        if self.ema_step % self.ema_update_rate == 0:
            if self.ema_step < self.ema_start:
                self._ema_denoise_model.load_state_dict(self._denoise_model.state_dict())
            else:
                self.ema.update_model_average(self._ema_denoise_model, self._denoise_model)
        return
    
    def _register_mdm_features(self, frame_components):
        """
        Registration of components and calculation of the dimensions of each chosen component
        """

        self._frame_components = []
        idx = 0
        self._feature_slices = dict()

        for comp in frame_components:
            comp = MDMFrameType[comp]
            self._frame_components.append(comp)
            if comp == MDMFrameType.ROOT_POS:
                dim = 3    
                
            elif comp == MDMFrameType.ROOT_ROT:
                dim = self._rot_changer.get_root_rot_dim()
               
            elif comp == MDMFrameType.JOINT_POS:
                dim = 3 * self._kin_char_model.get_num_non_root_joints()
               
            elif comp == MDMFrameType.JOINT_ROT:
                dim = self._rot_changer.get_joint_rot_dim()
               
            elif comp == MDMFrameType.JOINT_VEL:
                dim = 3 * self._kin_char_model.get_num_non_root_joints()

            elif comp == MDMFrameType.CONTACTS:
                if hasattr(self, "_custom_num_contacts"):
                    num_rb = self._custom_num_contacts
                else:
                    num_rb = self._kin_char_model.get_num_joints()
                dim = num_rb
            else:
                raise NameError(comp) 

            self._feature_slices[comp] = slice(idx, idx + dim)
            idx += dim
        self._motion_frame_dof = idx
        return
    
    def standardize_features(self, mdm_features):
        # Assuming mdm_features is of shape: [batch_size, seq_len, d]
        seq_len = mdm_features.shape[1]
        return (mdm_features - self._mdm_features_mean[:seq_len, :]) / self._mdm_features_std[:seq_len, :]
    
    def unstandardize_features(self, mdm_features):
        # Assuming mdm_features is of shape: [batch_size, seq_len, d]
        seq_len = mdm_features.shape[1]
        return mdm_features * self._mdm_features_std[:seq_len, :] + self._mdm_features_mean[:seq_len, :]
    
    def normalize_hf(self, hf):
        return hf / self._max_h
    
    def unnormalize_hf(self, hf):
        return hf * self._max_h
    
    def assemble_mdm_features(self, ml_component_dict: dict, standardize=True):
        """
        Assembling MDM feature from a dictionary of MotionLib components
        Input:
            ml_component_dict: dict()
        Output:
            mdm_features: [B x seq x mdm_dim]
        """

        mdm_feature_list = []
        # Order of components followed user specs in self._frame_components
        # start & end index of each component computed in self._register_mdm_feature, 
        # which will be called when this sampler class is created 
        for comp_name in self._frame_components:
            if comp_name == MDMFrameType.ROOT_POS:
                root_pos = ml_component_dict[MDMFrameType.ROOT_POS]
                mdm_feature_list.append(root_pos)
                
            elif comp_name == MDMFrameType.ROOT_ROT:
                ml_root_rot = ml_component_dict[MDMFrameType.ROOT_ROT]
                mdm_root_rot = self._rot_changer.convert_quat_to_rot_type(ml_root_rot)
                mdm_feature_list.append(mdm_root_rot)
               
            elif comp_name == MDMFrameType.JOINT_POS:
                ml_joint_pos = ml_component_dict[MDMFrameType.JOINT_POS]
                batch_size, seq_len, num_joints, d = ml_joint_pos.shape
                mdm_joint_pos = ml_joint_pos.view(batch_size, seq_len, -1)
                mdm_feature_list.append(mdm_joint_pos)

            elif comp_name == MDMFrameType.JOINT_ROT:
                ml_joint_rot = ml_component_dict[MDMFrameType.JOINT_ROT]
                # shape: [batch_size, seq_len, num_joints, 4]
                # new shape: [batch_size, seq_len, -1]
                mdm_joint_rot = self._rot_changer.convert_joint_quats_to_rot_type(ml_joint_rot)
                mdm_feature_list.append(mdm_joint_rot)

            elif comp_name == MDMFrameType.JOINT_VEL:
                ml_joint_vel = ml_component_dict[MDMFrameType.JOINT_VEL]
                mdm_feature_list.append(ml_joint_vel)

            elif comp_name == MDMFrameType.CONTACTS:
                ml_contacts = ml_component_dict[MDMFrameType.CONTACTS]
                mdm_feature_list.append(ml_contacts)
            else:
                raise NameError(comp_name) 
        mdm_features = torch.cat(mdm_feature_list, dim=-1)

        if standardize:
            mdm_features = self.standardize_features(mdm_features)

        return mdm_features
    
    def extract_motion_features(self, mdm_features: torch.Tensor, unstandardize=True):
        """
        Input:
            mdm_features: shape[batch_size, seq_len, motion_frame_dof]
        Outuput:
            ml_component_dict: dictionary of individual motion frame components
        """

        if unstandardize:
            mdm_features = self.unstandardize_features(mdm_features)

        ml_component_dict = dict()
        for i in range(len(self._frame_components)):
            comp_name = self._frame_components[i]
            curr_slice = self._feature_slices[comp_name]
            comp = mdm_features[..., curr_slice]
            if comp_name == MDMFrameType.ROOT_POS:
                root_pos = comp
                ml_component_dict[comp_name] = root_pos
            
            elif comp_name == MDMFrameType.ROOT_ROT:
                root_rot = comp
                root_rot = self._rot_changer.convert_rot_type_to_quat(root_rot)
                ml_component_dict[comp_name] = root_rot

            elif comp_name == MDMFrameType.JOINT_POS:
                joint_pos = comp
                batch_size, seq_len, d = joint_pos.shape
                num_joints = self._kin_char_model.get_num_non_root_joints()
                joint_pos = joint_pos.view(batch_size, seq_len, num_joints, 3)
                ml_component_dict[comp_name] = joint_pos

            elif comp_name == MDMFrameType.JOINT_ROT:
                joint_rot = comp
                joint_rot = self._rot_changer.convert_joint_rots_to_quat(joint_rot)
                ml_component_dict[comp_name] = joint_rot

            elif comp_name == MDMFrameType.JOINT_VEL:
                assert False, "Not implemented yet"

            elif comp_name == MDMFrameType.CONTACTS:
                contacts = comp
                ml_component_dict[comp_name] = contacts

        return ml_component_dict
    
    def compute_stats(self, sampler: MotionSampler, stats_filepath=None):
        # get every motion feature vector from all motions
        # then compute mean and std
        print("Computing motion stats...")


        stats_dir_name = os.path.dirname(self._cfg["sampler_save_filepath"])
        stats_file_name = os.path.splitext(os.path.basename(self._cfg["sampler_save_filepath"]))[0]
        if stats_filepath is None:
            stats_filepath = os.path.join(stats_dir_name, stats_file_name + "_stats.txt")
        
        if stats_filepath is None:
            stats_filepath = os.path.join(stats_dir_name, stats_file_name + "_stats.txt")
        
        if os.path.exists(stats_filepath):
            with open(stats_filepath, "r") as f:
                mdm_feature_stats = yaml.safe_load(f)
            mdm_features_mean = mdm_feature_stats["mean"]
            self._mdm_features_mean = torch.tensor(mdm_features_mean, device=self._device, dtype=torch.float32)
            
            mdm_features_std = mdm_feature_stats["std"]
            self._mdm_features_std = torch.tensor(mdm_features_std, device=self._device, dtype=torch.float32)
            print("Loaded stats from:", stats_filepath)
            return

        start_time = time.time()
        mdm_features_mean = torch.zeros(size=[self._seq_len, self._motion_frame_dof], dtype=torch.float32, device=self._device)
        mdm_features_std = torch.zeros_like(mdm_features_mean)

        num_sequences = 0
        print("Computing mean...")
        for id in sampler._mlib._motion_ids:
            motion_data = sampler.get_motion_sequences_for_id(id)
            mdm_features = self.assemble_mdm_features(motion_data, standardize=False)
            
            num_sequences += mdm_features.shape[0]
            mdm_features_mean += mdm_features.sum(dim=0)
        mdm_features_mean /= num_sequences

        print("Computing std...")
        for id in sampler._mlib._motion_ids:
            motion_data = sampler.get_motion_sequences_for_id(id)
            mdm_features = self.assemble_mdm_features(motion_data, standardize=False)
            mdm_features_std += torch.square(mdm_features - mdm_features_mean.unsqueeze(0)).sum(dim=0)
        mdm_features_std /= (num_sequences - 1)
        mdm_features_std = torch.sqrt(mdm_features_std)

        # Certain features should not be standardized when they are very simple, like contacts (1 or 0)
        if MDMFrameType.CONTACTS in self._feature_slices:
            mdm_features_mean[:, self._feature_slices[MDMFrameType.CONTACTS]] = 0.0
            mdm_features_std[:, self._feature_slices[MDMFrameType.CONTACTS]] = 1.0

        # Also std should never be too small
        mdm_features_std = torch.clamp(mdm_features_std, min=1e-5)

        print("mdm features mean:", mdm_features_mean)
        print("mdm features std:", mdm_features_std)

        end_time = time.time()
        print("computed stats in", end_time-start_time, "seconds.")

        self._mdm_features_mean = mdm_features_mean
        self._mdm_features_std = mdm_features_std

        mdm_feature_stats = {
            "mean": mdm_features_mean.cpu().numpy().tolist(),
            "std": mdm_features_std.cpu().numpy().tolist()
        }
        with open(stats_filepath, "w") as f:
            yaml.safe_dump(mdm_feature_stats, f)
        print("Saved stats to:", stats_filepath)
        return
    
    def construct_conds(self, motion_data: dict, hfs: torch.Tensor, target_info: TargetInfo,
                        test: bool):

        motion_features = self.assemble_mdm_features(motion_data)

        batch_size = motion_features.shape[0]
        device = motion_features.device

        conds = dict()
        true_prev_state = motion_features[:, 0:self._num_prev_states]

        conds[MDMKeyType.PREV_STATE_KEY] = true_prev_state
        if not test: # sample dropout masks in training mode
            rand_batch_prev_state = torch.rand(size=(batch_size,), dtype=torch.float32, device=device) < self._prev_state_attention_dropout
            conds[MDMKeyType.PREV_STATE_FLAG_KEY] = ~rand_batch_prev_state

            if self._use_target_obs:
                rand_batch_target = torch.rand(size=(batch_size,), dtype=torch.float32, device=device) < self._target_dropout
                conds[MDMKeyType.TARGET_FLAG_KEY] = ~rand_batch_target

            if self._use_heightmap_obs:
                rand_batch_obs = torch.rand(size=(batch_size,), dtype=torch.float32, device=device) < self._obs_dropout
                conds[MDMKeyType.OBS_FLAG_KEY] = ~rand_batch_obs

        else: # turning off dropout when in test mode
            ones = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
            conds[MDMKeyType.PREV_STATE_FLAG_KEY] = ones
            if self._use_target_obs:
                conds[MDMKeyType.TARGET_FLAG_KEY] = ones.clone() # probably don't need to clone...
            if self._use_heightmap_obs:
                conds[MDMKeyType.OBS_FLAG_KEY] = ones.clone()
        
        if self._use_heightmap_obs:
            conds[MDMKeyType.OBS_KEY] = self.normalize_hf(hfs)

        if self._use_target_obs:
            if self._target_type == TargetType.XY_DIR:
                target_dir = get_dir_from_canonicalized_pos_and_rot(target_info.future_pos, target_info.future_rot, 
                                                                self._target_dir_len_eps, self._target_dir_heading_eps)
                conds[MDMKeyType.TARGET_KEY] = target_dir.unsqueeze(1) # unsqueeze sequence dimension
            elif self._target_type == TargetType.XY_POS_AND_HEADING:

                # Get the last position of the motion
                last_pos = motion_features[:, -1:, 0:2].detach().clone()
                last_heading = torch_util.calc_heading(motion_data[MDMFrameType.ROOT_ROT][:, -1:, :]).unsqueeze(-1) / 3.14159

                target_cond = torch.cat([last_pos, last_heading], dim=-1)
                conds[MDMKeyType.TARGET_KEY] = target_cond
            else:
                assert False, "unsupported target type"

        return conds, motion_features
    
    def init_char_point_samples(self):
        char_point_samples = geom_util.get_minimal_char_point_samples(self._kin_char_model)
        self._char_point_samples = char_point_samples
        return

    def motion_difference(self, true_motion_data, pred_motion_data, conds, loss_fn):
        ## Computes the loss between true samples and generated samples
        # Uses quat diff angle since exp map differences are not accurate
        
        # Ultimately, we want our loss to be between the final generated motions,
        # not the difference between their compact representations

        #batch_size = x_0.shape[0]
        dt = 1.0 / self._sequence_fps
        losses = dict()

        root_pos_true = true_motion_data[MDMFrameType.ROOT_POS]
        root_pos_pred = pred_motion_data[MDMFrameType.ROOT_POS]
        root_rot_quat_true = true_motion_data[MDMFrameType.ROOT_ROT]
        root_rot_quat_pred = pred_motion_data[MDMFrameType.ROOT_ROT]
        joint_rot_quat_true = true_motion_data[MDMFrameType.JOINT_ROT]
        joint_rot_quat_pred = pred_motion_data[MDMFrameType.JOINT_ROT]

        body_pos_pred, body_rot_pred = self._kin_char_model.forward_kinematics(root_pos_pred, 
                                                                               root_rot_quat_pred, 
                                                                               joint_rot_quat_pred)

        body_pos_true, body_rot_true = self._kin_char_model.forward_kinematics(root_pos_true, 
                                                                                root_rot_quat_true, 
                                                                                joint_rot_quat_true)

        ## VELOCITY LOSS
        if self._use_vel_loss:
            root_vel_true = (root_pos_true[..., 1:, :] - root_pos_true[..., :-1, :]) / dt
            root_vel_pred = (root_pos_pred[..., 1:, :] - root_pos_pred[..., :-1, :]) / dt
            root_vel_loss = loss_fn(root_vel_true, root_vel_pred)

            root_drot_true = torch_util.quat_diff(root_rot_quat_true[..., :-1, :], root_rot_quat_true[..., 1:, :])
            root_drot_pred = torch_util.quat_diff(root_rot_quat_pred[..., :-1, :], root_rot_quat_pred[..., 1:, :])
            root_ang_vel_true = torch_util.quat_to_exp_map_grad_friendly(root_drot_true) / dt
            root_ang_vel_pred = torch_util.quat_to_exp_map_grad_friendly(root_drot_pred) / dt
            root_ang_vel_loss = loss_fn(root_ang_vel_true, root_ang_vel_pred)

            joint_drot_true = torch_util.quat_diff(joint_rot_quat_true[..., :-1, :, :], joint_rot_quat_true[..., 1:, :, :])
            joint_drot_pred = torch_util.quat_diff(joint_rot_quat_pred[..., :-1, :, :], joint_rot_quat_pred[..., 1:, :, :])
            joint_ang_vel_true = torch_util.quat_to_exp_map_grad_friendly(joint_drot_true) / dt
            joint_ang_vel_pred = torch_util.quat_to_exp_map_grad_friendly(joint_drot_pred) / dt
            joint_ang_vel_loss = loss_fn(joint_ang_vel_true, joint_ang_vel_pred)


            losses[LossType.VEL_ROOT_POS_LOSS] = self._w_vel_root_pos * root_vel_loss
            losses[LossType.VEL_ROOT_ROT_LOSS] = self._w_vel_root_rot * root_ang_vel_loss
            losses[LossType.VEL_JOINT_ROT_LOSS] = self._w_vel_joint_rot * joint_ang_vel_loss
            
        # SIMPLE LOSS (contact labels included here)
        if self._use_simple_loss:
            root_pos_loss = loss_fn(root_pos_true, root_pos_pred)
            root_rot_loss = loss_fn(torch_util.quat_diff_angle(root_rot_quat_true, root_rot_quat_pred), 0.0)
            joint_rot_loss = loss_fn(torch_util.quat_diff_angle(joint_rot_quat_true, joint_rot_quat_pred), 0.0)

            losses[LossType.SIMPLE_ROOT_POS_LOSS] = self._w_simple_root_pos * root_pos_loss
            losses[LossType.SIMPLE_ROOT_ROT_LOSS] = self._w_simple_root_rot * root_rot_loss
            losses[LossType.SIMPLE_JOINT_ROT_LOSS] = self._w_simple_joint_rot * joint_rot_loss


            if MDMFrameType.CONTACTS in true_motion_data:
                contacts_true = true_motion_data[MDMFrameType.CONTACTS]
                contacts_pred = pred_motion_data[MDMFrameType.CONTACTS]
                contact_loss = loss_fn(contacts_true, contacts_pred)
                losses[LossType.SIMPLE_CONTACT_LOSS] = self._w_simple_contacts * contact_loss

            if MDMFrameType.JOINT_POS in true_motion_data:
                simple_body_pos_pred = pred_motion_data[MDMFrameType.JOINT_POS]

                # ignore root body pos
                simple_body_pos_loss = loss_fn(body_pos_true[..., 1:, :], simple_body_pos_pred)
                losses[LossType.SIMPLE_BODY_POS_LOSS] = self._w_simple_body_pos * simple_body_pos_loss

        if self._use_pos_loss:
            # fk_body_pos_loss = loss_fn(body_pos_true[..., self._key_body_ids, :], 
            #                            body_pos_pred[..., self._key_body_ids, :])
            # angle_diff = torch_util.quat_diff_angle(body_rot_true[..., self._key_body_ids, :], 
            #                                         body_rot_pred[..., self._key_body_ids, :])
            fk_body_pos_loss = loss_fn(body_pos_true, body_pos_pred)
            angle_diff = torch_util.quat_diff_angle(body_rot_true, body_rot_pred)
            fk_body_rot_loss = loss_fn(angle_diff, 0.0)
            losses[LossType.FK_BODY_POS_LOSS] = self._w_body_pos * fk_body_pos_loss
            losses[LossType.FK_BODY_ROT_LOSS] = self._w_body_rot * fk_body_rot_loss

            if MDMFrameType.JOINT_POS in true_motion_data:
                body_pos_c_loss = loss_fn(body_pos_pred[..., 1:, :], simple_body_pos_pred)
                losses[LossType.BODY_POS_CONSISTENCY_LOSS] = self._w_body_pos_consistency * body_pos_c_loss


            if self._use_vel_loss:
                body_vel_true = (body_pos_true[:, 1:] - body_pos_true[:, :-1]) / dt
                body_vel_pred = (body_pos_pred[:, 1:] - body_pos_pred[:, :-1]) / dt
                fk_body_vel_loss = loss_fn(body_vel_true, body_vel_pred)
                losses[LossType.VEL_FK_BODY_POS_LOSS] = self._w_body_vel * fk_body_vel_loss


        if self._use_hf_collision_loss:
            hfs = self.unnormalize_hf(conds[MDMKeyType.OBS_KEY])

            point_samples = self._char_point_samples
            sdf = self.compute_point_hf_sdf(body_pos_pred, body_rot_pred, point_samples, hfs, base_z=self._min_h - 5.0)
            # shape: [batch_size, num_points]
            
            hf_loss = 0.5 * torch.sum(torch.square(torch.clamp(sdf, max=0.0)), dim=-1)
            losses[LossType.HF_COLLISION_LOSS] = self._w_hf * hf_loss

        if self._use_target_loss:
            if self._target_type == TargetType.XY_DIR:
                target_dir = conds[MDMKeyType.TARGET_KEY].squeeze(1)
                unnorm_pred_dir = root_pos_pred[:, -1, 0:2] - root_pos_pred[:, self._num_prev_states-1, 0:2]

                target_loss = torch.sum(-target_dir * unnorm_pred_dir, dim=-1)
                # give slight reward when dot product is moving forward up to our direction length eps
                target_loss = torch.clamp(target_loss, min=-self._target_dir_len_eps)

                losses[LossType.TARGET_LOSS] = target_loss * self._w_target
            else:
                losses[LossType.TARGET_LOSS] = 0.0
            
        return losses
    
    def _predict_x_0_training(self, x_0, conds, max_t, test, batch_size):
        if not test or self._test_mode == GenerationMode.NONE:
            t = torch.randint(0, max_t, size=(batch_size,1,1,), device=self._device)
            x_t = self.forward_diffusion(x_0, t)

            ind = torch.rand(size=(batch_size,), device=self._device) < self._prev_state_noise_ind_chance
            conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY] = ind

            predicted_x_0 = self._denoise_model(x_t, conds, t)
        else:
            ind = torch.ones(size=(batch_size,), device=self._device).to(dtype=torch.bool)
            conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY] = ind
            if self._test_mode == GenerationMode.MODE_DDIM:
                predicted_x_0 = self.ddim_inference(conds, stride=self._test_ddim_stride)
            elif self._test_mode == GenerationMode.MODE_REVERSE_DIFFUSION:
                predicted_x_0 = self.reverse_diffusion(conds)
            else:
                assert False
        return predicted_x_0
    
    def sample_motions(self, motion_sampler: MotionSampler, batch_size):
        motion_ids = motion_sampler._mlib.sample_motions(batch_size)

        start_times = motion_sampler._sample_motion_start_times(motion_ids,
                                                                self._sequence_duration)

        motion_data, hfs, target_info = motion_sampler.sample_motion_data(
            motion_ids=motion_ids, 
            motion_start_times=start_times)

        hfs = hfs.unsqueeze(1) # unsqueeze sequence dimension

        return motion_data, hfs, target_info

    def sample_and_compute_losses(self, max_t, motion_sampler, batch_size, loss_fn,
                                get_info=False, test=False,
                                ret_extra=False):
        # x_0 is sampled from the dataset.
        # To test our denoising model,
        # we want to noise x_0 by t diffusion timesteps,
        # then denoise it back t timesteps.
        # We compute the loss between the noised->denoised sample and the original sample x_0
        # note: we do not touch the first state in the sequence x_0, that is the prev_state

        # Let D be our motion difference funtion.
        # Then this function computes:
        # L = E_{x_t ~ G, t ~ (1, T)}[D(x_0, x_t)]

        with torch.no_grad():
            motion_data, hfs, target_info = self.sample_motions(motion_sampler, batch_size)

            conds, x_0 = self.construct_conds(motion_data, hfs, target_info, test=test)

        # select random timesteps
        predicted_x_0 = self._predict_x_0_training(x_0, conds, max_t, test, batch_size)

        predicted_motion_data = self.extract_motion_features(predicted_x_0)

        losses = self.motion_difference(motion_data, predicted_motion_data, conds, loss_fn)

        if ret_extra:
            return losses, hfs, x_0, predicted_x_0, conds
        else:
            return losses

    def sample_and_compute_loss(self, max_t, motion_sampler, batch_size, loss_fn,
                                get_info=False, test=False):

        losses = self.sample_and_compute_losses(max_t, motion_sampler, batch_size, loss_fn, get_info, test)
        
        # first make them means (so loss is averaged across batch dimension)
        for key in losses:
            losses[key] = losses[key].mean()

        loss = 0.0
        for key in losses:
            loss += losses[key]
        
        if get_info:
            return loss, losses
        else:
            return loss
    
    def forward_diffusion(self, x_0, t, noise = None):
        ## AKA forward diffusion process

        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff = self._diffusion_rates.sqrt_alphas_cumprod[t]
        std_coeff = self._diffusion_rates.sqrt_one_minus_alphas_cumprod[t]
        corrupted_mean = mean_coeff * x_0
        corrupted_deviation = std_coeff * noise
        corrupted_sample = corrupted_mean + corrupted_deviation

        return corrupted_sample
    
    def compute_point_hf_sdf(self, body_pos, body_rot, body_points, hfs, base_z = -10.0):
        # hfs are always in the character's local frame

        # body_pos: (batch_size, seq_len, num_bodies, 3)
        # body_rot: (batch_size, seq_len, num_bodies, 4)
        # body_points: (num_bodies, num_points(b), 3)
        # hfs: (batch_size, seq_len, dim_x, dim_y)

        # 1. apply body transformations to body_points
        # 2. get points in coord frame of hf (they already will be, no need)
        #       - apply root heading inverse
        #       - translate by mid point of hfs (which is (0, 0) so no need)
        # 3. call point hf sdf function to get sdfs
        # 4. loss is sum of squares of negative sdfs
        
        # NOTE: this function assumes base_z is lower than the minimum heightfield height

        batch_size = body_pos.shape[0]

        num_bodies = self._kin_char_model.get_num_joints()
        assert num_bodies == len(body_points)

        # NOTE: an alternative to cat is to pass body points as shape: (num_points, 3),
        # and keeping track of the slices for each body
        localized_points = []
        for b in range(num_bodies):
            # unsqueeze sequence dimension and batch dimension
            curr_body_points = body_points[b].unsqueeze(0).unsqueeze(0) # shape: (1, 1, num_points(b), 3)

            body_rot_unsq = body_rot[..., b, :].unsqueeze(2) # shape: (batch_size, seq_len, 1, 4)
            body_pos_unsq = body_pos[..., b, :].unsqueeze(2) # shape: (batch_size, seq_len, 1, 3)
            curr_body_points = torch_util.quat_rotate(body_rot_unsq, curr_body_points) + body_pos_unsq
            # shape: (batch_size, seq_len, num_points(b), 3)

            # append flattened points across seq_len and num_points dims
            localized_points.append(curr_body_points.view(batch_size, -1, 3))
        localized_points = torch.cat(localized_points, dim=1) # shape: (batch_size, num points, 3)

        negative_dims = torch.tensor([self._num_x_neg, self._num_y_neg], dtype=torch.float32, device=self._device)
        min_box_center = -self._dxdy * negative_dims
        min_box_center = min_box_center.unsqueeze(0).expand(batch_size, 2)
        sdf = terrain_util.points_hf_sdf(localized_points, hfs.squeeze(1), min_box_center, self._dxdy, base_z = base_z)
        return sdf # shape: [batch_size, num_points]
    
    
    
    def predict_x0(self, x_t, conds, t, cond_tokens, key_padding_mask):
        use_cfg = MDMKeyType.GUIDANCE_PARAMS in conds and conds[MDMKeyType.GUIDANCE_PARAMS].obs_cfg_scale is not None
        with torch.no_grad():
            if use_cfg:

                conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY][:] = True
                conds[MDMKeyType.PREV_STATE_FLAG_KEY][:] = True
                x_0 = self._denoise_model.fast_forward(x_t, conds, t, cond_tokens, key_padding_mask)
                conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY][:] = False
                conds[MDMKeyType.PREV_STATE_FLAG_KEY][:] = False
                x_0_no_prev_state = self._denoise_model.forward(x_t, conds, t)# cond_tokens, key_padding_mask)
                cfg_scale = conds[MDMKeyType.GUIDANCE_PARAMS].obs_cfg_scale
                predicted_x_0 = x_0 + cfg_scale * (x_0_no_prev_state - x_0)
            else:
                predicted_x_0 = self._denoise_model.fast_forward(x_t, conds, t, cond_tokens, key_padding_mask)

        return predicted_x_0
    
    #@torch.no_grad()
    def reverse_diffusion(self,
                          conds,
                          noise = None, 
                          start_timestep = None,
                          end_timestep = None, 
                          keep_all_samples = False,
                          stride=1):
        ## AKA reverse diffusion process

        batch_size = conds[MDMKeyType.PREV_STATE_KEY].shape[0]
        gen_seq_shape = (batch_size, self._seq_len, self._motion_frame_dof)
        if noise is None:
            x_t = torch.randn(gen_seq_shape, device=self._device)
        else:
            x_t = noise

        if start_timestep is None:
            start_timestep = self._diffusion_timesteps

        if end_timestep is None:
            end_timestep = 0

        if keep_all_samples:
            reverse_diffusion_samples = []
            reverse_diffusion_samples.append(x_t)

        key_padding_mask = self._denoise_model.create_key_padding_mask(batch_size, self._device)
        cond_tokens = self._denoise_model.embed_conds(conds, key_padding_mask)

        for t in reversed(range(end_timestep, start_timestep, stride)):
            t_torch = t * torch.ones(size=(batch_size, 1, 1), dtype=torch.int64, device=self._device)
            x_t = self.denoise_one_timestep(x_t, conds, t_torch, cond_tokens, key_padding_mask)

            if keep_all_samples:
                reverse_diffusion_samples.append(x_t)
        
        if keep_all_samples:
            return reverse_diffusion_samples
        else:
            return x_t
    
    #@torch.no_grad()
    def denoise_one_timestep_predict_x0(self, x_t, conds, t, cond_tokens, key_padding_mask):
        # Given x_t and t, denoise by one timestep to receive x_t-1.
        # This is done by using our network to predict the clean sample x_0,
        # then noising it back to x_{t-1} using the analytic formula
        # for q(x_{t-1} | x_t, x_0)
        # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process

        predicted_x_0 = self.predict_x0(x_t, conds, t, cond_tokens, key_padding_mask)
        predicted_x_0 = self.project_dofs(predicted_x_0)

        # NOTE: guidance is applied to clean signal before the noise happens again


        # NOTE: In our convention, we have "x_-1" being the final sample?
        if t[0, 0, 0] == 0: # no more corruption at the last reverse diffusion timestep
            return predicted_x_0
        else:
            # obtain mean as a linear combination of predicted_x_0 and xt
            # then get q(x_{t-1} | x_t, x_0(pred))
            mean_coeff1 = self._diffusion_rates.posterior_mean_coef1[t]
            mean_coeff2 = self._diffusion_rates.posterior_mean_coef2[t]
            posterior_std = self._diffusion_rates.posterior_std[t]
            noise = torch.randn_like(x_t, device=self._device)
            predicted_xt_minus_1 = mean_coeff1 * predicted_x_0 + mean_coeff2 * x_t + posterior_std * noise

            return predicted_xt_minus_1
    
    def denoise_one_timestep(self, x_t, conds, t, cond_tokens, key_padding_mask):
        if self._predict_mode == PredictMode.PREDICT_X0:
            predicted_xt_minus_1 = self.denoise_one_timestep_predict_x0(x_t, conds, t, cond_tokens, key_padding_mask)
        else:
            assert False
        return predicted_xt_minus_1
        
    #@torch.no_grad()
    def ddim_inference(self, 
                       conds,
                       noise = None, 
                       stride = 2, 
                       keep_all_samples = False):

        batch_size = conds[MDMKeyType.PREV_STATE_KEY].shape[0]
        gen_seq_shape = (batch_size, self._seq_len, self._motion_frame_dof)
        if noise is None:
            x_t = torch.randn(gen_seq_shape, device=self._device)
        else:
            x_t = noise

        if keep_all_samples:
            reverse_diffusion_samples = []
            reverse_diffusion_samples.append(x_t)

        if "timesteps" in conds:
            timesteps = conds["timesteps"]
        else:
            timesteps = reversed(range(0, self._diffusion_timesteps, stride))

        key_padding_mask = self._denoise_model.create_key_padding_mask(batch_size, self._device)
        cond_tokens = self._denoise_model.embed_conds(conds, key_padding_mask)

        for t in timesteps:
            t_torch = t * torch.ones(size=(batch_size, 1, 1), dtype=torch.int64, device=self._device)
            next_t_torch = t_torch - stride
            x_t = self.ddim_inference_step(x_t, conds, t_torch, next_t_torch, cond_tokens, key_padding_mask)

            if keep_all_samples:
                reverse_diffusion_samples.append(x_t)
        
        if keep_all_samples:
            return reverse_diffusion_samples
        else:
            return x_t
    
    #@torch.no_grad()
    def ddim_inference_step(self, x_t, conds, t, next_t, cond_tokens, key_padding_mask):
        predicted_x_0 = self.predict_x0(x_t, conds, t, cond_tokens, key_padding_mask)
        predicted_x_0 = self.project_dofs(predicted_x_0)

        # NOTE: In our convention, we have "x_-1" being the final sample?
        if t[0, 0, 0] == 0: # no more corruption at the last reverse diffusion timestep
            return predicted_x_0
        else:
            # obtain mean as a linear combination of predicted_x_0 and xt
            # then get q(x_{t-1} | x_t, x_0(pred))

            numer = self._diffusion_rates.sqrt_alphas_cumprod[t] * self._diffusion_rates.sqrt_one_minus_alphas_cumprod[next_t]
            denom = self._diffusion_rates.sqrt_one_minus_alphas_cumprod[t]
            mean_coeff_x_0 = self._diffusion_rates.sqrt_alphas_cumprod[next_t] - numer / denom 
            mean_coeff_x_t = self._diffusion_rates.sqrt_one_minus_alphas_cumprod[next_t] / self._diffusion_rates.sqrt_one_minus_alphas_cumprod[t]
            
            predicted_x_next_t = mean_coeff_x_0 * predicted_x_0 + mean_coeff_x_t * x_t
            return predicted_x_next_t
        
    def project_dofs(self, x):
        # If we are predicting joint positions and joint rotations, we will use FK
        # to get better joint positions
        motion_features = self.extract_motion_features(x)

        if MDMFrameType.JOINT_POS in self._frame_components and MDMFrameType.JOINT_ROT in self._frame_components:    
            root_pos = motion_features[MDMFrameType.ROOT_POS]
            root_rot = motion_features[MDMFrameType.ROOT_ROT]
            joint_rot = motion_features[MDMFrameType.JOINT_ROT]
            joint_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            motion_features[MDMFrameType.JOINT_POS] = joint_pos[..., 1:, :]

        if MDMFrameType.CONTACTS in self._frame_components:
            contacts = motion_features[MDMFrameType.CONTACTS]
            contacts = torch.clamp(contacts, min = 0.0, max = 1.0)
            motion_features[MDMFrameType.CONTACTS] = contacts

        x = self.assemble_mdm_features(motion_features)
        return x
    
    def train(self, motion_sampler, checkpoint_dir = None, test_only=False, stats_filepath=None):
        max_t = self._diffusion_timesteps
        if self._cfg["train_loss_fn"] == "squared_l2":
            train_loss_fn = squared_l2_loss_fn
        elif self._cfg["train_loss_fn"] == "pseudo_huber":
            train_loss_fn = pseudo_huber_loss_fn
        else:
            assert False
        if self._cfg["test_loss_fn"] == "squared_l2":
            test_loss_fn = squared_l2_loss_fn
        elif self._cfg["test_loss_fn"] == "pseudo_huber":
            test_loss_fn = pseudo_huber_loss_fn
        else:
            assert False

        self.compute_stats(motion_sampler, stats_filepath=stats_filepath)

        def log_losses(losses, loss, epoch, start_time, num_samples, log_to_wandb):
            self._logger.log("epoch", epoch)
            
            wall_time = (time.time() - start_time) / (60 * 60) # store time in hours
            self._logger.log("wall time", wall_time)
            self._logger.log("num samples", num_samples)

            self._logger.log("training loss", loss)
            for key in losses:
                name = "TRAIN " + LossType(key).name
                self._logger.log(LossType(key).name, losses[key].item())
            
            self._logger.print_log()

            if log_to_wandb:
                log_for_wandb = {"training loss": train_loss,
                                "wall time": wall_time}
                for key in losses:
                    name = "TRAIN " + LossType(key).name
                    log_for_wandb[name] = losses[key].item()

                log_for_wandb["num samples"] = num_samples
                wandb.log(log_for_wandb, step=num_samples)
            return

        start_time = time.time()

        # Initial loss
        with torch.no_grad():
            self._denoise_model.eval()
            initial_loss, initial_losses = self.sample_and_compute_loss(
                max_t, motion_sampler, self._test_size, 
                loss_fn=test_loss_fn, get_info=True)
            
            log_losses(initial_losses, initial_loss.item(), -1, start_time, 0, False)
        print("initial loss =", initial_loss.item())

        num_samples = 0

        # WEIGHTED SAMPLING WITH REPLACEMENT
        for epoch in range(self._epochs):
            running_loss_sum = 0.0
            running_losses_sum = {}
            num_running_losses = 0
            for i in range(self._iters_per_epoch):
                self._denoise_model.train()
                self._optimizer.zero_grad()
                loss, losses = self.sample_and_compute_loss(
                    max_t,
                    motion_sampler,
                    self._batch_size,
                    loss_fn=train_loss_fn,
                    get_info=True,
                    test=False,
                )
                running_loss_sum += loss.item()
                for key, value in losses.items():
                    if key not in running_losses_sum:
                        running_losses_sum[key] = 0.0
                    running_losses_sum[key] += value.item()
                num_running_losses += 1

                if not test_only:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._denoise_model.parameters(), 1.0)
                    self._optimizer.step()
                    # update ema model if activated
                    with torch.no_grad():
                        self.update()

            num_samples += self._batch_size * self._iters_per_epoch

            running_loss_avg = running_loss_sum / num_running_losses
            train_losses = {
                key: torch.tensor(value / num_running_losses)
                for key, value in running_losses_sum.items()
            }
            train_loss = running_loss_avg

            log_losses(train_losses, train_loss, epoch, start_time, num_samples, True)

            if not test_only:
                if epoch % self._epochs_per_checkpoint == 0 and epoch > 0 and checkpoint_dir is not None:
                    checkpoint_save_path = checkpoint_dir / ("model_" + str(epoch) + ".ckpt")
                    self.save_checkpoint(checkpoint_save_path)

        self._denoise_model.eval()
        final_loss = self.sample_and_compute_loss(max_t, motion_sampler, self._test_size, loss_fn=test_loss_fn).item()
        print("final loss =", final_loss)
        return
    
    ## INTERFACE WITH IG ENV FUNCTION
    def gen_sequence(self, conds, ddim_stride=None, mode=GenerationMode.MODE_DDIM):
        self._denoise_model.eval()
        
        if self._use_heightmap_obs:
            if len(conds[MDMKeyType.OBS_KEY].shape) == 3: # (num_samples, dim_x, dim_y)
                # -> (num_samples, 1 (for sequence dim), dim_x, dim_y)
                normalized_hf_obs = torch.clamp(conds[MDMKeyType.OBS_KEY].unsqueeze(dim=1), min=self._min_h, max=self._max_h)
                normalized_hf_obs = self.normalize_hf(normalized_hf_obs)
                conds[MDMKeyType.OBS_KEY] = normalized_hf_obs

        input_motion_data = conds[MDMKeyType.PREV_STATE_KEY]

        # motion_data dict will be passed into prev_state_key
        prev_state = self.assemble_mdm_features(input_motion_data)
        conds[MDMKeyType.PREV_STATE_KEY] = prev_state
        if len(conds[MDMKeyType.PREV_STATE_KEY].shape) == 2:
            conds[MDMKeyType.PREV_STATE_KEY] = conds[MDMKeyType.PREV_STATE_KEY].unsqueeze(dim=1)


        if self._use_target_obs:
            if len(conds[MDMKeyType.TARGET_KEY].shape) == 2:
                # unsqueeze sequence dimension
                conds[MDMKeyType.TARGET_KEY] = conds[MDMKeyType.TARGET_KEY].unsqueeze(dim=1)

        if mode==GenerationMode.MODE_DDIM:
            assert (ddim_stride is not None)
            diffusion_sample = self.ddim_inference(conds=conds, stride=ddim_stride)
        elif mode==GenerationMode.MODE_REVERSE_DIFFUSION:
            # compute sample
            diffusion_timesteps = self._diffusion_timesteps if "timesteps" not in conds else conds["timesteps"]
            diffusion_sample = self.reverse_diffusion(conds=conds, start_timestep=diffusion_timesteps)
        else:
            assert False

        no_noise_ind = conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY].unsqueeze(dim=-1).unsqueeze(dim=-1)
        out_prev_state = torch.where(no_noise_ind, prev_state, diffusion_sample[:, 0:self._num_prev_states])
        
        final_sequence = torch.cat([out_prev_state, diffusion_sample[:, self._num_prev_states:, :]], dim = 1)                
        output_motion_data = self.extract_motion_features(final_sequence)

        return output_motion_data
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            "cfg": self._cfg,
            "model": self._denoise_model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

        if getattr(self, "use_ema", False) and hasattr(self, "_ema_denoise_model"):
            checkpoint["ema_model"] = self._ema_denoise_model.state_dict()
            checkpoint["ema_meta"] = {
                "ema_step": getattr(self, "ema_step", None),
                "ema_decay": getattr(self, "ema_decay", None),
                "ema_start": getattr(self, "ema_start", None),
                "ema_update_rate": getattr(self, "ema_update_rate", None),
            }

        # 🔑 Save all member variables that are tensors / numpy arrays / numbers
        extra_vars = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                extra_vars[k] = v.detach().cpu()  # keep portable
            elif isinstance(v, np.ndarray):
                extra_vars[k] = v
            elif isinstance(v, (int, float, bool, str)):
                extra_vars[k] = v
        checkpoint["extra"] = extra_vars

        torch.save(checkpoint, path)
        return
    
    @classmethod
    def load_checkpoint(cls, path: str, device="cpu", use_ema_model=False):
        checkpoint_path = path_loader.resolve_path(path)

        if not checkpoint_path.is_file():
            assert False, f"Checkpoint file does not exist: {checkpoint_path}"

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        cfg = checkpoint["cfg"]
        cfg['device'] = device

        obj = cls(cfg)
        obj._denoise_model.load_state_dict(checkpoint["model"])
        obj._optimizer.load_state_dict(checkpoint["optimizer"])

        if "ema_model" in checkpoint:
            obj._ema_denoise_model.load_state_dict(checkpoint["ema_model"])
            for k, v in checkpoint.get("ema_meta", {}).items():
                setattr(obj, k, v)

        # 🔑 Restore extra vars
        for k, v in checkpoint.get("extra", {}).items():
            # move tensors back to device
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            setattr(obj, k, v)

        obj._device = device

        if use_ema_model and obj.use_ema:
            print("Using ema model...")
            obj._denoise_model = obj._ema_denoise_model

        return obj