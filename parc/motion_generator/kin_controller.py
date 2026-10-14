from typing import Optional

import torch

import parc.anim.kin_char_model as kin_char_model
import parc.motion_generator.gen_util as gen_util
import parc.motion_generator.mdm as mdm
import parc.util.terrain_util as terrain_util
from parc.util.motion_util import MotionFrames


class KinematicController:
    
    def __init__(self,
                 mdm_model: mdm.MDM,
                 char_model: kin_char_model.KinCharModel,
                 initial_frames: MotionFrames):
        
        self._mdm_model = mdm_model
        self._char_model = char_model

        self._curr_frame = 0
        self._replan_period = 5
        self._batch_size = 1
        self._motion_frames = initial_frames
        self._paused = False

        return
    
    def set_frames(self, motion_frames: MotionFrames):
        self._motion_frames = motion_frames
        return

    def step(self,
             target_world_pos: Optional[torch.Tensor],
             terrain: terrain_util.SubTerrain,
             mdm_settings: gen_util.MDMGenSettings,
             target_dir: Optional[torch.Tensor] = None
             ):
        
        if self._paused:
            return

        self._curr_frame += 1

        if self._curr_frame > self._replan_period:
            if len(target_world_pos.shape) == 1:
                target_world_pos = target_world_pos.unsqueeze(0).expand(self._batch_size, -1)
                
            self._motion_frames = gen_util.gen_mdm_motion(
                target_world_pos = target_world_pos,
                prev_frames = self._motion_frames.get_slice(slice(self._replan_period-1, self._replan_period+1)),
                terrain = terrain,
                mdm_model = self._mdm_model,
                char_model = self._char_model,
                mdm_settings = mdm_settings,
                target_dir = target_dir)
            
            self._curr_frame = 2 # start rendering the earliest generated future frame
            

        return
    
    def get_curr_motion_frame(self):
        return self._motion_frames.get_slice(self._curr_frame)