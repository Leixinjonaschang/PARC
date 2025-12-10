import os
import sys
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path

# Make sure PARC modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parc_root = os.path.abspath(os.path.join(current_dir, "../../"))
if parc_root not in sys.path:
    sys.path.insert(0, parc_root)

# Import PARC modules
import anim.kin_char_model as kin_char_model
import util.terrain_util as terrain_util
import util.torch_util as torch_util
import util.geom_util as geom_util
import zmotion_editing_tools.motion_edit_lib as medit_lib
import tools.motion_opt.motion_optimization as moopt

# Body name mappings for different robot models
# Maps standard names -> robot-specific names
BODY_NAME_MAPPINGS = {
    "unitree_g1": {
        "left_foot": "left_ankle_roll_link",
        "right_foot": "right_ankle_roll_link",
        # Note: left_rubber_hand/right_rubber_hand are commented out in g1_mocap_29dof.xml
        # The actual hand end-effector bodies are the wrist_yaw links
        "left_hand": "left_wrist_yaw_link",
        "right_hand": "right_wrist_yaw_link",
    },
    # Add more robot mappings as needed
}

def compute_foot_contacts_generic(motion_frames, char_model, fps=30.0, terrain=None, 
                                  contact_eps=0.02, contact_vel_thresh=1.0,
                                  left_foot_name="left_foot", right_foot_name="right_foot"):
    """
    Compute foot contacts with configurable body names and velocity filtering.
    
    Contact is detected when:
    1. Foot height < contact_eps (or penetrating)
    2. Foot speed < contact_vel_thresh (to avoid false positives during swing phase)
    
    Args:
        motion_frames: Motion data tensor
        char_model: KinCharModel instance
        fps: Frames per second of the motion data
        terrain: SubTerrain instance (optional).
        contact_eps: Height threshold (default 0.02m).
        contact_vel_thresh: Velocity threshold (default 1.0 m/s). 
                            Set to large value to disable velocity check.
        left_foot_name: Name of left foot body
        right_foot_name: Name of right foot body
    
    Returns:
        contacts: Tensor of shape (num_frames, num_bodies)
    """
    import anim.motion_lib as motion_lib
    
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)
    
    # Compute body velocities using finite difference
    # Shape: (num_frames, num_bodies, 3)
    # Pad the last frame with zero velocity or replicate previous velocity
    if num_frames > 1:
        body_vel = torch.zeros_like(body_pos)
        body_vel[:-1] = (body_pos[1:] - body_pos[:-1]) * fps
        body_vel[-1] = body_vel[-2] # Replicate last valid velocity
    else:
        body_vel = torch.zeros_like(body_pos)

    # Try to get body IDs with fallback
    try:
        lf_id = char_model.get_body_id(left_foot_name)
    except:
        print(f"Warning: Could not find body '{left_foot_name}', skipping left foot contacts")
        lf_id = None
        
    try:
        rf_id = char_model.get_body_id(right_foot_name)
    except:
        print(f"Warning: Could not find body '{right_foot_name}', skipping right foot contacts")
        rf_id = None

    key_ids = [id for id in [lf_id, rf_id] if id is not None]

    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]
        key_body_vel = body_vel[:, body_id]
        
        # Calculate speed (magnitude of velocity)
        key_body_speed = torch.norm(key_body_vel, dim=-1)

        # Get geom for this body
        if body_id < len(char_model._geoms) and len(char_model._geoms[body_id]) > 0:
            geom = char_model._geoms[body_id][0]
            geom_offset = geom._offset
            geom_dims = geom._dims
            
            box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
            min_height = torch.min(box_points[..., 2], dim=-1).values
            # Use foot xy position for terrain height lookup
            foot_xy = key_body_pos[..., :2]
        else:
            # Fallback: use body position directly
            min_height = key_body_pos[..., 2]
            foot_xy = key_body_pos[..., :2]

        # Determine ground height at each foot position
        if terrain is not None:
            # Query terrain height at foot xy positions using SubTerrain's method
            terrain_height = terrain.get_hf_val_from_points(foot_xy)
        else:
            # Assume flat ground at z=0
            terrain_height = torch.zeros_like(min_height)
        
        # Height above terrain surface (negative means penetration)
        height_above_terrain = min_height - terrain_height
        
        # Contact logic:
        # 1. Height check: close to ground OR penetrating
        is_close_to_ground = height_above_terrain < contact_eps
        
        # 2. Velocity check: moving slowly
        is_moving_slowly = key_body_speed < contact_vel_thresh
        
        # Combine checks
        # Note: We might want to be lenient with velocity if there is significant penetration,
        # but generally sliding contacts are bad for training anyway.
        contact_mask = is_close_to_ground & is_moving_slowly
        
        contacts[:, body_id] = contact_mask.float()

    return contacts


def compute_body_constraints_generic(root_pos, root_rot, joint_rot, contacts,
                                     char_model, terrain, robot_type=None):
    """
    Compute body constraints with configurable body names for different robots.
    This wraps the original compute_approx_body_constraints with body name mapping.
    """
    # Get body name mapping
    left_foot_name = "left_foot"
    right_foot_name = "right_foot"
    left_hand_name = "left_hand"
    right_hand_name = "right_hand"
    
    if robot_type and robot_type in BODY_NAME_MAPPINGS:
        mapping = BODY_NAME_MAPPINGS[robot_type]
        left_foot_name = mapping.get("left_foot", left_foot_name)
        right_foot_name = mapping.get("right_foot", right_foot_name)
        left_hand_name = mapping.get("left_hand", left_hand_name)
        right_hand_name = mapping.get("right_hand", right_hand_name)
    
    # Get body IDs
    def get_body_id_safe(name):
        try:
            return char_model.get_body_id(name)
        except:
            print(f"Warning: Body '{name}' not found, skipping constraints for this body")
            return None
    
    lf_id = get_body_id_safe(left_foot_name)
    rf_id = get_body_id_safe(right_foot_name)
    lh_id = get_body_id_safe(left_hand_name)
    rh_id = get_body_id_safe(right_hand_name)
    
    # Forward kinematics
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    
    # Initialize empty body constraints
    body_constraints = [[] for _ in range(len(char_model._body_names))]
    
    # Helper functions (copied from motion_optimization.py)
    def extract_consecutive_trues(bool_tensor):
        bool_tensor = bool_tensor.flatten()
        indices = torch.nonzero(bool_tensor, as_tuple=True)[0]
        if len(indices) == 0:
            return []
        diff = indices[1:] - indices[:-1]
        breaks = torch.nonzero(diff > 1, as_tuple=True)[0] + 1
        breaks = [0] + breaks.tolist()
        split_indices = []
        for i in range(len(breaks)-1):
            split_indices.append(indices[breaks[i]:breaks[i+1]].clone())
        if breaks[-1] < indices.shape[0] - 1:
            split_indices.append(indices[breaks[-1]:indices.shape[0]].clone())
        return split_indices
    
    def estimate_approx_contact_points(body_contacts, body_positions, contact_threshold=0.9):
        body_contacts = body_contacts > contact_threshold
        contact_frame_ids = extract_consecutive_trues(body_contacts)
        contact_points = []
        for group in contact_frame_ids:
            avg_pos = torch.mean(body_positions[group], axis=0)
            contact_points.append(avg_pos)
        if len(contact_points) > 0:
            contact_points = torch.stack(contact_points)
        else:
            contact_points = torch.tensor([], device=body_contacts.device)
        return contact_points, contact_frame_ids
    
    def create_opt_constraints(points, contact_frame_ids):
        constraint_list = []
        for i in range(len(contact_frame_ids)):
            # BodyConstraint uses class attributes, not constructor args
            constraint = moopt.BodyConstraint()
            constraint.start_frame_idx = contact_frame_ids[i][0].item()
            constraint.end_frame_idx = contact_frame_ids[i][-1].item()
            constraint.constraint_point = points[i]
            constraint_list.append(constraint)
        return constraint_list
    
    # Process each body that was found
    for body_id, body_name in [(lf_id, "left_foot"), (rf_id, "right_foot"), 
                                (lh_id, "left_hand"), (rh_id, "right_hand")]:
        if body_id is None:
            continue
        
        body_positions = body_pos[:, body_id]
        body_contacts = contacts[:, body_id]
        
        points, contact_frame_ids = estimate_approx_contact_points(body_contacts, body_positions)
        
        if len(points) > 0:
            body_constraints[body_id] = create_opt_constraints(points, contact_frame_ids)
    
    return body_constraints


def apply_height_fix(frames, contacts, char_model, terrain=None, num_fix_frames=10, left_foot_name="left_foot", right_foot_name="right_foot"):
    """
    Apply height correction to the motion based on initial foot contacts.
    Calculates the average offset needed to bring contacting feet to the TERRAIN surface
    during the first few frames.
    
    Args:
        frames: Motion frames tensor (modified in-place)
        contacts: Contact tensor
        char_model: KinCharModel
        terrain: SubTerrain instance (optional). If None, assumes flat ground at z=0.
        num_fix_frames: Number of initial frames to use for calculation
        left_foot_name: Name of left foot body
        right_foot_name: Name of right foot body
    """
    import anim.motion_lib as motion_lib
    
    device = frames.device
    num_frames = frames.shape[0]
    fix_frames = min(num_frames, num_fix_frames)
    
    # Get foot body IDs
    try:
        lf_id = char_model.get_body_id(left_foot_name)
        rf_id = char_model.get_body_id(right_foot_name)
    except:
        print("Warning: Could not find foot bodies for height fix. Skipping.")
        return

    # Extract pose for first N frames
    subset_frames = frames[:fix_frames]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(subset_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)
    
    # FK to get foot positions
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)
    
    # Collect foot relative heights (foot_z - terrain_z) where contact == 1
    height_diffs = []
    
    # Heuristic for foot bottom offset: ankle joint is ~5cm above ground
    FOOT_BOTTOM_OFFSET = 0.05 
    
    for body_id in [lf_id, rf_id]:
        # For mesh-based robots like G1, accessing geom[0] (the visual mesh) 
        # to compute bounding box is unreliable and can yield huge offsets (e.g. -1.0m).
        # It's safer to use the Joint Position directly and subtract a fixed offset.
        
        # Use body (joint) position
        joint_z = body_pos[:, body_id, 2]
        foot_xy = body_pos[:, body_id, :2]
        
        # Estimate foot bottom Z
        min_height = joint_z - FOOT_BOTTOM_OFFSET
        
        # Check contact state (using the provided contacts tensor)
        is_contact = contacts[:fix_frames, body_id] > 0.5
        
        if is_contact.any():
            # Get actual foot Z
            contact_foot_z = min_height[is_contact]
            
            # Get terrain Z at foot XY
            contact_foot_xy = foot_xy[is_contact]
            
            if terrain is not None:
                terrain_z = terrain.get_hf_val_from_points(contact_foot_xy)
            else:
                terrain_z = torch.zeros_like(contact_foot_z)
                
            # Diff = Foot - Terrain
            # We want this to be 0
            diff = contact_foot_z - terrain_z
            height_diffs.append(diff)
            
            # # DEBUG INFO
            # if len(height_diffs) == 1: # Print debug for first contacting body
            #     print(f"    [DEBUG] Body {body_id}:")
            #     print(f"      Joint Z:      {joint_z[is_contact][:3].cpu().numpy()}")
            #     print(f"      Foot Z (est): {contact_foot_z[:3].cpu().numpy()}")
            #     print(f"      Terrain Z:    {terrain_z[:3].cpu().numpy()}")
            #     print(f"      Diff:         {diff[:3].cpu().numpy()}")
            
    if len(height_diffs) > 0:
        all_diffs = torch.cat(height_diffs)
        # Calculate mean difference
        avg_diff = torch.mean(all_diffs)
        
        # Correction offset is negative of the difference
        # If avg_diff is +0.1 (floating 10cm above terrain), offset is -0.1
        offset = -avg_diff
        
        print(f"  Height Fix: Detected avg foot-terrain diff {avg_diff:.4f}m. Applying offset {offset:.4f}m to all frames.")
        
        # Apply offset to Root Z (index 2)
        frames[..., 2] += offset
        
        # Verify fix
        new_z = frames[..., 2]
        # print(f"    [DEBUG] Root Z before: {subset_frames[0, 2]:.4f}, after: {new_z[0]:.4f}")
    else:
        print("  Height Fix: No foot contacts detected in first 10 frames. Skipping correction.")


def process_robot_motion(input_path, output_path, robot_model_path, 
                         terrain_type="flat", 
                         contact_eps=0.02,
                         contact_vel_thresh=1.0, # Added velocity threshold parameter
                         device="cpu",
                         robot_type=None,
                         terrain_source_path=None):
    """
    Process retargeted robot motion data (containing only frames & fps) to generate
    contacts, body constraints, and other necessary fields for PARC training.
    
    Args:
        input_path: Path to the input .pkl file (frames, fps)
        output_path: Path to save the processed .pkl file
        robot_model_path: Path to the robot's XML/URDF file for kinematics
        terrain_type: Type of terrain to assume ("flat" or "reconstruct")
        contact_eps: Height threshold for foot contact detection
        contact_vel_thresh: Velocity threshold for foot contact detection
        device: Computing device ("cpu" or "cuda")
        robot_type: Robot type for body name mapping (e.g., "unitree_g1")
        terrain_source_path: Optional path to a .pkl file containing the terrain object.
                             If provided, terrain will be loaded from here instead of input_path.
    """
    
    print(f"Processing {input_path}...")
    print(f"Using robot model: {robot_model_path}")
    
    # 1. Load Robot Model
    try:
        char_model = kin_char_model.KinCharModel(device)
        char_model.load_char_file(str(robot_model_path))
    except Exception as e:
        print(f"Error loading robot model: {e}")
        return

    # 2. Load Motion Data
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
        
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Validate input data
    if 'frames' not in data:
        print(f"Error: 'frames' key missing in {input_path}")
        return
        
    # Convert frames to torch tensor
    frames = data['frames']
    if isinstance(frames, np.ndarray):
        frames = torch.tensor(frames, dtype=torch.float32, device=device)
    elif isinstance(frames, list):
        frames = torch.tensor(np.array(frames), dtype=torch.float32, device=device)
    
    fps = data.get('fps', 30) # Default to 30 if missing
    
    # 3. Setup Terrain and Contacts from Source
    # Check if we should load terrain/contacts from an external source file
    terrain = None
    source_contacts = None
    
    if terrain_source_path and os.path.exists(terrain_source_path):
        try:
            print(f"Loading data from external source: {terrain_source_path}")
            with open(terrain_source_path, 'rb') as f:
                src_data = pickle.load(f)
            
            # Load Terrain
            if 'terrain' in src_data:
                terrain = src_data['terrain']
                print("  Successfully loaded terrain from external source.")
            else:
                print(f"  Warning: 'terrain' key not found in {terrain_source_path}")
                
            # Load Contacts (if available)
            if 'contacts' in src_data:
                source_contacts = src_data['contacts']
                print("  Successfully loaded contacts from external source.")
            else:
                print(f"  Warning: 'contacts' key not found in {terrain_source_path}")
                
        except Exception as e:
            print(f"  Error loading external source data: {e}")

    # Fallback to internal terrain if not loaded externally
    if terrain is None:
        if 'terrain' in data:
            terrain = data['terrain']
        else:
            print(f"Warning: 'terrain' key missing in {input_path} and no external source provided. Terrain generation/contact check assumes flat ground.")

    # Process terrain object
    if terrain is not None:
        # First convert numpy arrays to torch tensors if needed
        if hasattr(terrain, 'to_torch'):
            terrain.to_torch(device)
        # Handle old terrain format if update_old method exists
        if hasattr(terrain, 'update_old'):
            terrain.update_old()
    
    # 4. Compute Contacts
    print("Computing contacts...")
    
    # Determine body names based on robot type
    left_foot_name = "left_foot"
    right_foot_name = "right_foot"
    
    # Determine indices for mapping source contacts
    # We need to handle different source data structures.
    # Case 1: Source is standard Humanoid (15 bodies) from PARC/DeepMimic
    # Case 2: Source is simplified 4-limb contact data
    
    contacts = None
    
    if robot_type and robot_type in BODY_NAME_MAPPINGS:
        mapping = BODY_NAME_MAPPINGS[robot_type]
        left_foot_name = mapping.get("left_foot", left_foot_name)
        right_foot_name = mapping.get("right_foot", right_foot_name)
        # Add hands if available in mapping
        left_hand_name = mapping.get("left_hand", "left_hand")
        right_hand_name = mapping.get("right_hand", "right_hand")
        print(f"Using body mapping for {robot_type}")
        print(f"  Left Foot  -> {left_foot_name}")
        print(f"  Right Foot -> {right_foot_name}")
        print(f"  Left Hand  -> {left_hand_name}")
        print(f"  Right Hand -> {right_hand_name}")

        # If we have source contacts, try to map them first
        if source_contacts is not None:
            try:
                # Convert source contacts to tensor if needed
                if isinstance(source_contacts, np.ndarray):
                    src_contacts_t = torch.tensor(source_contacts, dtype=torch.float32, device=device)
                else:
                    src_contacts_t = source_contacts.to(device)
                
                # Check shapes
                num_src_cols = src_contacts_t.shape[1]
                print(f"  Source contact columns: {num_src_cols}")
                
                if src_contacts_t.shape[0] != frames.shape[0]:
                    print(f"  Warning: Source contacts frame count ({src_contacts_t.shape[0]}) does not match motion ({frames.shape[0]}). Skipping copy.")
                elif num_src_cols < 2:
                    print(f"  Warning: Source contacts have too few bodies ({num_src_cols}). Skipping copy.")
                else:
                    print("  Mapping source contacts to target robot...")
                    
                    num_frames = frames.shape[0]
                    num_bodies = len(char_model._body_names)
                    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)
                    
                    # Helper to safe map
                    def map_contact(src_idx, body_name):
                        if src_idx >= num_src_cols:
                            print(f"    Warning: Source index {src_idx} out of bounds (cols={num_src_cols}).")
                            return
                        try:
                            bid = char_model.get_body_id(body_name)
                            contacts[:, bid] = src_contacts_t[:, src_idx]
                            # print(f"    Mapped Source Col {src_idx} -> Target '{body_name}' (ID {bid})")
                        except:
                            print(f"    Warning: Target body '{body_name}' not found in robot model.")

                    # --- MAPPING LOGIC ---
                    
                    # Logic for Standard Humanoid (15 bodies)
                    # Based on standard XML order:
                    # 5: right_hand, 8: left_hand, 11: right_foot, 14: left_foot
                    if num_src_cols == 15:
                        print("  Detected 15-column source (Standard Humanoid). Using specific index mapping.")
                        map_contact(14, left_foot_name)   # Left Foot
                        map_contact(11, right_foot_name)  # Right Foot
                        map_contact(8, left_hand_name)    # Left Hand
                        map_contact(5, right_hand_name)   # Right Hand
                        
                    # Logic for simplified data (e.g. [LF, RF, LH, RH] or [LF, RF])
                    elif num_src_cols == 4:
                        print("  Detected 4-column source. Assuming [LF, RF, LH, RH] order.")
                        map_contact(0, left_foot_name)
                        map_contact(1, right_foot_name)
                        map_contact(2, left_hand_name)
                        map_contact(3, right_hand_name)
                        
                    elif num_src_cols == 2:
                        print("  Detected 2-column source. Assuming [LF, RF] order.")
                        map_contact(0, left_foot_name)
                        map_contact(1, right_foot_name)
                        
                    else:
                        print(f"  Warning: Unknown source contact column count ({num_src_cols}). Cannot infer mapping automatically.")
                        contacts = None # Fallback to compute
                        
                    if contacts is not None:
                        print("  Source contacts mapped successfully.")
                    
            except Exception as e:
                print(f"  Error mapping source contacts: {e}")
                contacts = None # Fallback to compute

    # Fallback to geometric computation if contacts are still None
    if contacts is None:
        print("  Using geometric contact detection (Fallback)...")
        try:
            contacts = compute_foot_contacts_generic(
                frames, 
                char_model, 
                fps=fps,
                terrain=terrain,  # Pass terrain for height-aware contact detection
                contact_eps=contact_eps,
                contact_vel_thresh=contact_vel_thresh,
                left_foot_name=left_foot_name,
                right_foot_name=right_foot_name
            )
        except Exception as e:
            print(f"Error computing contacts: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: create zero contacts
            num_frames = frames.shape[0]
            num_bodies = len(char_model._body_names)
            contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)
            print("Warning: Using zero contacts as fallback.")

    # 4.5 Apply Height Correction
    # If we have valid contacts (either from source or computed), we can correct the root height
    # so that "contacting" feet are actually on the ground (z=0).
    # This is critical for retargeted motions where floating/penetration is common.
    print("Applying height correction...")
    try:
        apply_height_fix(frames, contacts, char_model, 
                        terrain=terrain, # Pass terrain object here
                        num_fix_frames=10, 
                        left_foot_name=left_foot_name, 
                        right_foot_name=right_foot_name)
    except Exception as e:
        print(f"Error applying height fix: {e}")

    # 5. Compute Body Constraints (opt:body_constraints)
    print("Computing body constraints...")
    root_pos = frames[..., 0:3]
    root_rot = torch_util.exp_map_to_quat(frames[..., 3:6])
    joint_rot = char_model.dof_to_rot(frames[..., 6:])
    
    try:
        # compute_approx_body_constraints uses hardcoded body names
        # We need to temporarily add aliases if using a different robot
        body_constraints = compute_body_constraints_generic(
            root_pos=root_pos,
            root_rot=root_rot,
            joint_rot=joint_rot,
            contacts=contacts,
            char_model=char_model,
            terrain=terrain,
            robot_type=robot_type
        )
    except Exception as e:
        print(f"Error computing body constraints: {e}")
        import traceback
        traceback.print_exc()
        body_constraints = [[] for _ in range(len(char_model._body_names))]

    # 6. Compute Heightfield Mask Indices (hf_mask_inds)
    # This is used for terrain encoding during training.
    print("Computing hf_mask_inds...")
    
    # We need body sampling points for this.
    # Using geom_util.get_char_point_samples as seen in parc_2_kin_gen.py
    # Adjust sampling parameters as needed.
    try:
        body_points = geom_util.get_char_point_samples(
            char_model,
            sphere_num_subdivisions=0, # Standard low res for efficient collision check
            box_num_slices=2,
            box_dim_x=2,
            box_dim_y=2,
            capsule_num_circle_points=4,
            capsule_num_sphere_subdivisons=0,
            capsule_num_cylinder_slices=2
        )
        
        hf_mask_inds = terrain_util.compute_hf_extra_vals(
            motion_frames=frames, 
            terrain=terrain, 
            char_model=char_model, 
            char_body_points=body_points
        )
    except Exception as e:
        print(f"Error computing hf_mask_inds: {e}")
        hf_mask_inds = []

    # 7. Assemble and Save Data
    print("Saving processed data...")
    
    # Convert tensors back to numpy/python objects for pickling
    processed_data = {
        "frames": frames.cpu().numpy(),
        "fps": fps,
        "loop_mode": data.get("loop_mode", "CLAMP"),  # Preserve loop_mode or default to CLAMP
        "contacts": contacts.cpu().numpy(),
        "opt:body_constraints": body_constraints, # This is a list of custom objects, pickle handles it
        "hf_mask_inds": hf_mask_inds, # List of tensors usually, might need to be careful
        "terrain": terrain,  # terrain is already in torch format on the correct device
    }
    
    # Move terrain to CPU for saving
    if hasattr(terrain, 'to_torch'):
        terrain.to_torch("cpu")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Done. Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess robot motion data for PARC.")
    parser.add_argument("--input", required=True, help="Input .pkl file or directory containing .pkl files")
    parser.add_argument("--output", required=True, help="Output .pkl file or directory")
    parser.add_argument("--robot_model", required=True, help="Path to robot XML/URDF file")
    parser.add_argument("--robot_type", default=None, 
                        choices=["unitree_g1", "humanoid", None],
                        help="Robot type for body name mapping (e.g., 'unitree_g1')")
    parser.add_argument("--contact_eps", type=float, default=0.01, help="Height threshold for contact detection")
    parser.add_argument("--contact_vel_thresh", type=float, default=0.2, help="Velocity threshold for contact detection (m/s)")
    parser.add_argument("--device", default="cpu", help="Device to run computations on")
    parser.add_argument("--terrain_dir", default=None, help="Optional directory containing original motion files with terrain data")
    
    args = parser.parse_args()
    
    # Auto-detect robot type from model path if not specified
    robot_type = args.robot_type
    if robot_type is None:
        if "g1" in args.robot_model.lower() or "unitree_g1" in args.robot_model.lower():
            robot_type = "unitree_g1"
            print(f"Auto-detected robot type: {robot_type}")
    
    input_path = args.input
    output_path = args.output
    terrain_dir = args.terrain_dir
    
    def find_terrain_file(src_file, root_dir, terrain_root_dir, robot_type):
        """Helper to find matching terrain file."""
        if not terrain_root_dir:
            return None
            
        # Determine subdirectory structure
        if os.path.isdir(root_dir):
            rel_path = os.path.relpath(src_file, root_dir)
            sub_dir = os.path.dirname(rel_path)
            base_name = os.path.basename(src_file)
        else:
            sub_dir = ""
            base_name = os.path.basename(src_file)
            
        name, ext = os.path.splitext(base_name)
        candidates = [base_name]
        
        # Try stripping robot type suffix (e.g. "_to_unitree_g1")
        if robot_type:
            suffix = f"_to_{robot_type}"
            if name.endswith(suffix):
                orig_name = name[:-len(suffix)] + ext
                candidates.append(orig_name)
                
        # Search in corresponding subdirectory of terrain_dir
        search_dirs = [os.path.join(terrain_root_dir, sub_dir), terrain_root_dir]
        
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for cand in candidates:
                cand_path = os.path.join(d, cand)
                if os.path.exists(cand_path):
                    return cand_path
                    
        return None
    
    if os.path.isdir(input_path):
        # Batch process directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(".pkl"):
                    src_file = os.path.join(root, file)
                    
                    # Maintain directory structure
                    rel_path = os.path.relpath(src_file, input_path)
                    dest_file = os.path.join(output_path, rel_path)
                    
                    terrain_file = find_terrain_file(src_file, input_path, terrain_dir, robot_type)
                    
                    process_robot_motion(src_file, dest_file, args.robot_model, 
                                         contact_eps=args.contact_eps, 
                                         contact_vel_thresh=args.contact_vel_thresh,
                                         device=args.device,
                                         robot_type=robot_type,
                                         terrain_source_path=terrain_file)
    else:
        # Single file
        terrain_file = find_terrain_file(input_path, os.path.dirname(input_path), terrain_dir, robot_type)
        process_robot_motion(input_path, output_path, args.robot_model, 
                             contact_eps=args.contact_eps, 
                             contact_vel_thresh=args.contact_vel_thresh,
                             device=args.device,
                             robot_type=robot_type,
                             terrain_source_path=terrain_file)

if __name__ == "__main__":
    main()

