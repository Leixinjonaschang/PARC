import enum
from typing import List

import numpy as np
import polyscope as ps
import torch
import trimesh

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.motion_edit_lib as medit_lib
import parc.util.torch_util as torch_util
from parc.util.motion_util import MotionFrames


def create_sphere(center, radius, name="sphere", color=(1.0, 0.0, 0.0), num_subdivisions=0):
    sphere_mesh = trimesh.primitives.Sphere(radius=radius, subdivisions=num_subdivisions)
    ps_sphere_mesh = ps.register_surface_mesh(
        name=name, vertices=sphere_mesh.vertices, faces=sphere_mesh.faces,
        color=color
    )
    transform = ps_sphere_mesh.get_transform()
    transform[:3, 3] = center
    ps_sphere_mesh.set_transform(transform)
    
    return ps_sphere_mesh

def create_line(p0, p1, name="line", color=(1.0, 0.0, 0.0), radius=0.005):
    """
    Draw a single 3D line segment in Polyscope.

    Args:
        p0 (array-like): 3D coordinates of start point
        p1 (array-like): 3D coordinates of end point
        name (str): name of the line object in Polyscope
        color (tuple): RGB triple in [0,1]
        radius (float): line thickness (cylinder radius)
    """
    points = np.array([p0, p1])
    edges = np.array([[0, 1]])

    curve = ps.register_curve_network(name, points, edges, radius=radius)
    curve.set_color(color)
    return curve

def create_vector_mesh(vector, name="vector", radius=0.05, color=[1.0, 0.0, 0.0]) -> ps.SurfaceMesh:

    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)

    length = torch.norm(vector)
    assert length > 0.0

    z_axis = torch.tensor([0.0, 0.0, 1.0])

    vector_dir = vector / length
    vector_rot_quat = torch_util.quat_diff_vec(z_axis, vector_dir)
    vector_rot_mat = torch_util.quat_to_matrix(vector_rot_quat)
    transform = np.eye(4)
    transform[:3, :3] = vector_rot_mat.numpy()
    transform[:3, 3] = vector.numpy() / 2.0
    cylinder = trimesh.primitives.Cylinder(radius, length, transform)

    transform = np.eye(4)
    transform[:3, 3] = vector.numpy()
    transform[:3, :3] = vector_rot_mat.numpy()


    cone = trimesh.creation.cone(radius=radius*3.0, height = length*0.15, sections=8,
                                 transform=transform)


    vertices = np.concatenate([cylinder.vertices, cone.vertices], axis=0)
    faces = np.concatenate([cylinder.faces, cone.faces + cylinder.vertices.shape[0]], axis=0)

    ps_mesh = ps.register_surface_mesh(name, vertices, faces)
    ps_mesh.set_color(color)

    return ps_mesh

def create_orthogonal_axes_mesh(name):
    radius = 0.05
    length = 1.0
    head_radius_scale = 3.0
    head_fraction = 0.2
    head_height = length * head_fraction
    
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    x_axis = torch.tensor([1.0, 0.0, 0.0])
    x_axis_rot_quat = torch_util.quat_diff_vec(z_axis, x_axis)
    x_axis_rot_mat = torch_util.quat_to_matrix(x_axis_rot_quat)
    transform = np.eye(4)
    transform[:3, :3] = x_axis_rot_mat.numpy()
    transform[:3, 3] = [length * 0.5, 0.0, 0.0]
    x_axis_cyl = trimesh.primitives.Cylinder(radius, length, transform)

    transform = np.eye(4)
    transform[:3, :3] = x_axis_rot_mat.numpy()
    transform[:3, 3] = [length, 0.0, 0.0]
    x_axis_cone = trimesh.creation.cone(
        radius=radius * head_radius_scale,
        height=head_height,
        sections=12,
        transform=transform)

    y_axis = torch.tensor([0.0, 1.0, 0.0])
    y_axis_rot_quat = torch_util.quat_diff_vec(z_axis, y_axis)
    y_axis_rot_mat = torch_util.quat_to_matrix(y_axis_rot_quat)
    transform = np.eye(4)
    transform[:3, :3] = y_axis_rot_mat.numpy()
    transform[:3, 3] = [0.0, length * 0.5, 0.0]
    y_axis_cyl = trimesh.primitives.Cylinder(radius, length, transform)

    transform = np.eye(4)
    transform[:3, :3] = y_axis_rot_mat.numpy()
    transform[:3, 3] = [0.0, length, 0.0]
    y_axis_cone = trimesh.creation.cone(
        radius=radius * head_radius_scale,
        height=head_height,
        sections=12,
        transform=transform)

    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, length * 0.5]
    z_axis_cyl = trimesh.primitives.Cylinder(radius, length, transform)

    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, length]
    z_axis_cone = trimesh.creation.cone(
        radius=radius * head_radius_scale,
        height=head_height,
        sections=12,
        transform=transform)

    num_x_cyl_verts = x_axis_cyl.vertices.shape[0]
    num_x_cone_verts = x_axis_cone.vertices.shape[0]
    num_y_cyl_verts = y_axis_cyl.vertices.shape[0]
    num_y_cone_verts = y_axis_cone.vertices.shape[0]
    num_z_cyl_verts = z_axis_cyl.vertices.shape[0]
    num_z_cone_verts = z_axis_cone.vertices.shape[0]

    vertices = np.concatenate(
        [
            x_axis_cyl.vertices,
            x_axis_cone.vertices,
            y_axis_cyl.vertices,
            y_axis_cone.vertices,
            z_axis_cyl.vertices,
            z_axis_cone.vertices,
        ],
        axis=0,
    )

    faces = np.concatenate(
        [
            x_axis_cyl.faces,
            x_axis_cone.faces + num_x_cyl_verts,
            y_axis_cyl.faces + num_x_cyl_verts + num_x_cone_verts,
            y_axis_cone.faces + num_x_cyl_verts + num_x_cone_verts + num_y_cyl_verts,
            z_axis_cyl.faces + num_x_cyl_verts + num_x_cone_verts + num_y_cyl_verts + num_y_cone_verts,
            z_axis_cone.faces
            + num_x_cyl_verts
            + num_x_cone_verts
            + num_y_cyl_verts
            + num_y_cone_verts
            + num_z_cyl_verts,
        ],
        axis=0,
    )
    
    ps_mesh = ps.register_surface_mesh(name, vertices, faces)

    colors = np.zeros_like(vertices)
    x_end = num_x_cyl_verts + num_x_cone_verts
    y_start = x_end
    y_end = y_start + num_y_cyl_verts + num_y_cone_verts
    z_start = y_end

    colors[0:x_end, :] = [1.0, 0.0, 0.0]
    colors[y_start:y_end, :] = [0.0, 1.0, 0.0]
    colors[z_start:, :] = [0.0, 0.0, 1.0]
    ps_mesh.add_color_quantity("colors", colors, enabled=True)

    return ps_mesh

def create_origin_axis_mesh():
    return create_orthogonal_axes_mesh("origin axes")

def update_char_motion_mesh(body_pos, body_rot, char_bodies: List[ps.SurfaceMesh], char_model: kin_char_model.KinCharModel):
    for b in range(0, char_model.get_num_joints()):
        pose = np.eye(4)
        pose[:3, 3] = body_pos[b].cpu().numpy()

        rot_mat = torch_util.quat_to_matrix(body_rot[b]).cpu().numpy()
        pose[:3, :3] = rot_mat

        char_bodies[b].set_transform(pose)
    return

def create_char_trimeshes(char_model: kin_char_model.KinCharModel):
    device = char_model._device
    char_trimeshes = []
    #print("char_model num joints:", char_model.get_num_joints())
    for b in range(0, char_model.get_num_joints()):
        geoms = char_model.get_geoms(b)

        body_trimeshes = []
        for geom in geoms:
            #trimesh.primitives.Capsule()
            if geom._shape_type == kin_char_model.GeomType.SPHERE:
                r = geom._dims.item()
                offset = geom._offset.cpu().numpy()
                body = trimesh.primitives.Sphere(r, offset)
                body_trimeshes.append(body)        
            elif geom._shape_type == kin_char_model.GeomType.BOX:
                extents = geom._dims.cpu().numpy() * 2.0
                offset = geom._offset.cpu().numpy()
                transform = np.eye(4)
                transform[:3, 3] = offset

                if geom._quat is not None:
                    quat = torch.from_numpy(geom._quat).to(dtype=torch.float32, device="cpu")
                    transform[:3, :3] = torch_util.quat_to_matrix(quat).numpy()
                box = trimesh.primitives.Box(extents=extents, transform=transform)
                body_trimeshes.append(box)
            elif geom._shape_type == kin_char_model.GeomType.CAPSULE:
                fromto = geom._dims.cpu().numpy()
                offset = geom._offset.cpu().numpy() + fromto/2.0 # because capsules start at their centers
                transform = np.eye(4)
                transform[:3, 3] = offset

                # z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                # dim_axis = geom._dims / torch.norm(geom._dims)
                # axis = torch.cross(z_axis, dim_axis)
                # axis_norm = torch.norm(axis)
                # bad = axis_norm < 5e-1

                # # if char_model.get_num_joints() == 24:
                # #     print(axis_norm)
                # #     print(axis)

                # if bad:
                #     axis = z_axis
                # else:
                #     axis = axis / torch.norm(axis)

                
                
                # # Compute rotation angle using dot product
                # angle = torch.acos(torch.dot(axis, geom._dims))

                # rotation = torch_util.axis_angle_to_quat(axis, angle)

                # rotation = torch_util.quat_to_matrix(rotation)
                # transform[:3, :3] = rotation.cpu().numpy()
                h = np.linalg.norm(fromto)
                fromto = fromto / h
                z_axis = np.array([0, 0, 1])
                v = np.cross(z_axis, fromto)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, fromto)
                if s < 1e-8:
                    # Axis aligned, rotation not needed or 180 degrees
                    if c < 0:
                        # Opposite direction, 180° rotation
                        R = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
                    else:
                        R = np.eye(3)
                else:
                    vx = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
                transform[:3, :3] = R

                body = trimesh.primitives.Capsule(radius=geom._radius, height=h, transform=transform)

                transform=np.eye(4)
                quat = torch.from_numpy(geom._quat).to(dtype=torch.float32, device="cpu")
                transform[:3, :3] = torch_util.quat_to_matrix(quat).numpy()
                body.apply_transform(transform)

                body_trimeshes.append(body)
            elif geom._shape_type == kin_char_model.GeomType.CYLINDER:
                radius = geom._dims[0].item()
                half_height = geom._dims[1].item()
                height = 2 * half_height
                offset = geom._offset
                quat = torch.from_numpy(geom._quat).to(dtype=torch.float32, device="cpu")
                transform=np.eye(4)
                transform[:3, 3] = offset
                transform[:3, :3] = torch_util.quat_to_matrix(quat).numpy()
                body = trimesh.primitives.Cylinder(radius=radius, height=height, transform=transform)
                body_trimeshes.append(body)
            elif geom._shape_type == kin_char_model.GeomType.MESH:

                mesh_name = geom._mesh_name
                mesh = char_model._meshes[mesh_name].copy()
                #if geom._quat is not None:
                offset = geom._offset
                quat = torch.from_numpy(geom._quat).to(dtype=torch.float32, device="cpu")
                transform=np.eye(4)
                transform[:3, 3] = offset
                transform[:3, :3] = torch_util.quat_to_matrix(quat).numpy()
                mesh.apply_transform(transform)
                body_trimeshes.append(mesh)

        if len(geoms) == 0:
            # place a default sphere
            r = 0.01
            offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            body = trimesh.primitives.Sphere(r, offset)
            body_trimeshes.append(body)   
        char_trimeshes.append(body_trimeshes)

    return char_trimeshes

def create_char_mesh(char_name, color, transparency, char_model: kin_char_model.KinCharModel):
    char_trimeshes = create_char_trimeshes(char_model)
    char_bodies_ps = []
    for b in range(0, char_model.get_num_joints()):
        body_trimeshes = char_trimeshes[b]

        # combine all trimeshes for one body into one mesh
        vertices = []
        faces = []
        curr_num_verts = 0
        for i in range(len(body_trimeshes)):
            #mesh_name = "body " + str(b) + " geom " + str(i)
            #print(mesh_name)
            vertices.append(body_trimeshes[i].vertices)
            faces.append(body_trimeshes[i].faces + curr_num_verts)
            curr_num_verts += body_trimeshes[i].vertices.shape[0]
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)

        ps_mesh_name = char_name + "_" + char_model._body_names[b]
        ps_mesh = ps.register_surface_mesh(ps_mesh_name, vertices, faces)
        ps_mesh.set_color(color)
        ps_mesh.set_transparency(transparency)
        char_bodies_ps.append(ps_mesh)
    return char_bodies_ps

def compute_motion_frames(motion_times: torch.Tensor, mlib: motion_lib.MotionLib,
                          pos_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                          rot_offset = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)):
    device = mlib._device
    pos_offset = pos_offset.to(device=device)
    rot_offset = rot_offset.to(device=device)

    use_contact_info = mlib._contact_info
    motion_ids = torch.zeros_like(motion_times, dtype=torch.int64, device=mlib._device)

    if use_contact_info:
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = mlib.calc_motion_frame(motion_ids, motion_times)
    else:
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = mlib.calc_motion_frame(motion_ids, motion_times)
        contacts = None

    root_pos = torch_util.quat_rotate(rot_offset.unsqueeze(0), root_pos)
    root_pos = root_pos + pos_offset
    root_rot = torch_util.quat_multiply(rot_offset, root_rot)

    body_pos, body_rot = mlib._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    motion_frames = MotionFrames(
        root_pos=root_pos,
        root_rot=root_rot,
        joint_rot=joint_rot,
        body_pos=body_pos,
        body_rot=body_rot,
        contacts=contacts
    )
    return motion_frames



# Class that creates a sequence of static character state meshes, and concats them all into one mesh (to speed up rendering)
class MotionSequencePS:
    def __init__(self, name, start_color, end_color, num_frames, start_time, end_time, mlib: motion_lib.MotionLib, visibility=0.3,
                 frame_colors = None, position_based = False, root_pos_spacing = 0.75,
                 render_skeleton: bool = True, skeleton_radius: float = 0.01):
        self.name = name
        self.num_frames = num_frames
        self.char_model = mlib._kin_char_model
        self.render_skeleton = render_skeleton
        self.skeleton_radius = skeleton_radius

        start_color = np.array(start_color)
        end_color = np.array(end_color)

        vertices = []
        faces = []
        colors = []
        nodes = []
        edges = []
        node_colors = []
        num_verts = 0
        num_nodes = 0
        if not self.render_skeleton:
            char_trimeshes = create_char_trimeshes(self.char_model)
        #dt = (end_time - start_time) / (self.num_frames - 1.0)


        
        if not position_based:
            motion_times = torch.linspace(start=start_time, end=end_time, steps=num_frames, device=mlib._device, dtype=torch.float32)
        else:
            motion_times = self.position_based_motion_times(mlib, root_pos_spacing=root_pos_spacing)
            num_frames = motion_times.shape[0] 
        motion_frames = compute_motion_frames(motion_times, mlib)
        

        for i in range(num_frames):
            body_pos = motion_frames.body_pos[i]
            body_rot = motion_frames.body_rot[i]
            curr_color = (end_color - start_color) * i / (self.num_frames - 1.0) + start_color#-1.0) + start_color
            if frame_colors is not None:
                curr_color = frame_colors[i]
            
            if not self.render_skeleton:
                for b in range(self.char_model.get_num_joints()):
                    pos = body_pos[b].cpu().numpy()
                    rot = torch_util.quat_to_matrix(body_rot[b]).cpu().numpy()

                    trimeshes = char_trimeshes[b]
                    for geom in range(len(trimeshes)):
                        mesh = trimeshes[geom]
                        new_verts = mesh.vertices @ rot.transpose() + pos
                        vertices.append(new_verts)
                        faces.append(mesh.faces + num_verts)
                        num_verts += mesh.vertices.shape[0]
                        new_colors = np.zeros_like(new_verts)
                        new_colors[:] = curr_color
                        colors.append(new_colors)
            else:
                for b in range(self.char_model.get_num_joints()):
                    pos = body_pos[b].cpu().numpy()
                    nodes.append(pos)
                    node_colors.append(curr_color)

                for b in range(1, self.char_model.get_num_joints()):
                    parent_idx = self.char_model.get_parent_id(b)
                    if isinstance(parent_idx, torch.Tensor):
                        parent_idx = int(parent_idx.item())
                    else:
                        parent_idx = int(parent_idx)
                    if parent_idx >= 0:
                        edges.append([num_nodes + parent_idx, num_nodes + b])

                num_nodes += self.char_model.get_num_joints()

        if not self.render_skeleton:
            vertices = np.concatenate(vertices, axis=0)
            faces = np.concatenate(faces, axis = 0)
            colors = np.concatenate(colors, axis = 0)
            self.mesh = ps.register_surface_mesh(name, vertices, faces)
            self.mesh.add_color_quantity("colors", colors, enabled=True)
        else:
            nodes = np.asarray(nodes)
            edges = np.asarray(edges, dtype=int)
            if edges.ndim == 1:
                edges = edges.reshape((-1, 2))
            node_colors = np.asarray(node_colors)
            self.mesh = ps.register_curve_network(name, nodes, edges, radius=self.skeleton_radius)
            self.mesh.set_radius(self.skeleton_radius,relative=False)
            if node_colors.size > 0:
                self.mesh.add_color_quantity("colors", node_colors, enabled=True)

        if hasattr(self.mesh, "set_transparency"):
            self.mesh.set_transparency(visibility)
        self.visibility = visibility

        
        self.root_offset = np.array([0.0, 0.0, 0.0])
        self.root_rot = np.array([0.0, 0.0, 0.0, 1.0])
        return
    
    def position_based_motion_times(self, mlib: motion_lib.MotionLib, root_pos_spacing=0.3):

        root_positions = mlib._frame_root_pos
        num_frames = mlib._motion_num_frames[0].item()
        curr_frame_idx = 0
        next_frame_idx = 1

        vis_frame_idxs = [curr_frame_idx]
        while next_frame_idx < num_frames:
            
            curr_root_pos = root_positions[curr_frame_idx]
            next_root_pos = root_positions[next_frame_idx]
            dist = torch.linalg.norm(next_root_pos - curr_root_pos)

            if dist >= root_pos_spacing:
                vis_frame_idxs.append(next_frame_idx)
                curr_frame_idx = next_frame_idx
            next_frame_idx += 1

        motion_times = torch.tensor(vis_frame_idxs, dtype=torch.float32, device=mlib._device)
        motion_times = motion_times / mlib._motion_fps[0].item()
        print(motion_times)
        return motion_times
    
    def remove(self):
        self.mesh.remove()
        return
    
    def set_enabled(self, val):
        self.mesh.set_enabled(val)
        return
    
class CharacterPS:
    def __init__(self, name, color, char_model: kin_char_model.KinCharModel, history_length = 1):
        self.name = name
        #self.body_pos = None
        #self.body_rot = None
        self.motion_frames = None
        self.char_model = char_model
        self._history_length = history_length

        self.color = np.array(color)

        self.create_char_mesh()


        shadow = trimesh.primitives.Cylinder(0.5, 0.01)
        shadow_name = name + "_shadow"
        self.shadow = ps.register_surface_mesh(shadow_name, shadow.vertices, shadow.faces)
        self.shadow.set_color([color[0] * 0.1, color[1] * 0.1, color[2] * 0.1])

        self.pos_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.rot_offset = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        self.motion_frames = MotionFrames()
        self.motion_frames.init_blank_frames(char_model, history_length=1)
        self.motion_frames = self.motion_frames.squeeze(0)

        self.set_prev_state_enabled(False)
        return
    
    def create_char_mesh(self):
        self.meshes = []
        for i in range(self._history_length):
            curr_name = self.name + str(i)
            transparency = 0.5 + (i / self._history_length)
            if self._history_length == 1:
                transparency = 1.0
            curr_color = self.color * transparency
            self.meshes.append(create_char_mesh(curr_name, curr_color, transparency, self.char_model))
            
        # also create spheres for body points
        self.body_cube_meshes = []
        for b in range(0, self.char_model.get_num_joints()):
            mesh = trimesh.primitives.Box(extents=[0.05, 0.05, 0.05])
            surf_mesh = ps.register_surface_mesh(
                name = self.name + "_" + self.char_model.get_body_name(b) + "_body_point", 
                vertices=mesh.vertices, faces=mesh.faces,
                color=[0.0, 0.7, 0.0])
            self.body_cube_meshes.append(surf_mesh)
        self.set_body_points_enabled(False)
        return
    
    def get_prev_state_enabled(self):
        return self._prev_state_enabled

    def set_prev_state_enabled(self, val):
        if self._history_length > 1:
            for prev_frame_idx in range(self._history_length-1):
                for mesh in self.meshes[prev_frame_idx]:
                    mesh.set_enabled(val)
        self._prev_state_enabled = val
        return
    
    def set_to_time(self, motion_time, dt, mlib: motion_lib.MotionLib):
        self.motion_frames = []

        motion_times = []
        for i in range(self._history_length):
            t = motion_time - i * dt
            motion_times.append(t)
        motion_times.reverse()
        motion_times = torch.tensor(motion_times, dtype=torch.float32, device=mlib._device)
        motion_frames = compute_motion_frames(motion_times, mlib, 
                                              pos_offset=self.pos_offset, 
                                              rot_offset=self.rot_offset)

        self.motion_frames = motion_frames

        if mlib._contact_info:
            contacts = motion_frames.contacts

            self.update_contact_colours(contacts)
            
        return
    
    def update_contact_colours(self, contacts):
        # set color based on contact info
        red = np.array([1.0, 0.0, 0.0])

        for i in range(self._history_length):
            for contact_body_id in range(0, self.char_model.get_num_contact_bodies()):
                body_name = self.char_model.get_contact_body_name(contact_body_id)
                body_id = self.char_model.get_body_id(body_name)
                body_contact = contacts[i, contact_body_id].item()
                self.meshes[i][body_id].set_color(red * body_contact + self.color * (1.0 - body_contact))
        
        return
    
    def update_contact_colours_one_frame(self, contacts):
        assert len(contacts.shape) == 1
        red = np.array([1.0, 0.0, 0.0])
        for contact_body_id in range(0, self.char_model.get_num_contact_bodies()):
            body_name = self.char_model.get_contact_body_name(contact_body_id)
            body_id = self.char_model.get_body_id(body_name)
            body_contact = contacts[contact_body_id].item()
            self.meshes[-1][body_id].set_color(red * body_contact + self.color * (1.0 - body_contact))
        return
    
    def compute_motion_frames(self, motion_time, dt, seq_len, hist_len, mlib: motion_lib.MotionLib):
        motion_times = []
        for i in range(seq_len):
            t = motion_time + (i + 1.0 - hist_len) * dt
            motion_times.append(t)
        motion_times = torch.tensor(motion_times, dtype=torch.float32, device=mlib._device)
        motion_frames = compute_motion_frames(motion_times, mlib, 
                                            pos_offset=self.pos_offset, 
                                            rot_offset=self.rot_offset)
        return motion_frames
    
    def set_root_pos(self, root_pos: torch.Tensor):
        assert root_pos.shape == torch.Size([3])
        self.motion_frames.root_pos[-1] = root_pos
        return
    
    def set_root_rot_expmap(self, root_rot: torch.Tensor):
        assert root_rot.shape == torch.Size([3])
        root_rot_quat = torch_util.exp_map_to_quat(root_rot)
        self.motion_frames.root_rot[-1] = root_rot_quat
        return
    
    def set_root_rot_quat(self, root_rot: torch.Tensor):
        assert root_rot.shape == torch.Size([4])
        self.motion_frames.root_rot[-1] = root_rot
        return
    
    def set_joint_dofs(self, joint_dofs):
        self.motion_frames.joint_rot[-1] = self.char_model.dof_to_rot(joint_dofs)
        return
    
    def set_to_zero_pose(self):
        device=self.char_model._device
        self.set_root_pos(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device))
        self.set_root_rot_expmap(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device))
        zero_joint_dofs = torch.zeros(size=[self.char_model.get_dof_size()], dtype=torch.float32, device=device)
        self.set_joint_dofs(zero_joint_dofs)
        return
    
    def forward_kinematics(self, root_pos=None, root_rot=None, joint_rot=None, update_transforms=True, shadow_height=0.0):
        if root_pos is None:
            root_pos = self.motion_frames.root_pos[-1]
        if root_rot is None:
            root_rot = self.motion_frames.root_rot[-1]
        if joint_rot is None:
            joint_rot = self.motion_frames.joint_rot[-1]
        body_pos, body_rot = self.char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        self.motion_frames.body_pos[-1] = body_pos
        self.motion_frames.body_rot[-1] = body_rot
        if update_transforms:
            self.update_transforms(shadow_height)
        return
    
    def update_transforms(self, shadow_height=0.0):
        body_pos = self.motion_frames.body_pos
        body_rot = self.motion_frames.body_rot
        for i in range(self._history_length):
            update_char_motion_mesh(body_pos[i], body_rot[i], self.meshes[i], self.char_model)
        transform = np.eye(4)
        transform[:2, 3] += body_pos[-1][0, 0:2].cpu().numpy()
        transform[2, 3] = shadow_height
        self.shadow.set_transform(transform)

        for b in range(body_pos[-1].shape[0]):
            transform = np.eye(4)
            transform[:3, 3] = body_pos[-1][b].cpu().numpy()
            transform[:3, :3] = torch_util.quat_to_matrix(body_rot[-1][b]).cpu().numpy()
            self.body_cube_meshes[b].set_transform(transform)
        return
    
    def remove(self):
        for char_state_meshes in self.meshes:
            for ps_mesh in char_state_meshes:
                ps_mesh.remove()
        self.shadow.remove()
        return
    
    def set_color(self, val):
        self.color = val
        for char_state_meshes in self.meshes:
            for ps_mesh in char_state_meshes:
                ps_mesh.set_color(val)
        self.shadow.set_color([val[0] * 0.1, val[1] * 0.1, val[2] * 0.1])
        return
    
    def get_body_pos(self, body_id=None):
        if body_id is None:
            return self.motion_frames.body_pos[-1]
        else:
            return self.motion_frames.body_pos[-1][body_id]
    
    def get_body_rot(self, body_id=None):
        if body_id is None:
            return self.motion_frames.body_rot[-1]
        else:
            return self.motion_frames.body_rot[-1][body_id]
        
    def get_root_vel(self):
        return self.root_vel[-1]
        
    def get_root_ang_vel(self):
        return self.root_ang_vel[-1]
    
    def set_enabled(self, val):
        char_mesh = self.meshes[-1]
        for body_mesh in char_mesh:
            body_mesh.set_enabled(val)
        self.shadow.set_enabled(val)
        return
    
    def set_shadow_enabled(self, val):
        self.shadow.set_enabled(val)
        return
    
    def get_body_points_enabled(self):
        return self._body_points_enabled
    
    def set_body_points_enabled(self, val):
        for b in range(len(self.body_cube_meshes)):
            self.body_cube_meshes[b].set_enabled(val)
        self._body_points_enabled = val
        return
    
    def set_transparency(self, transparency, frame_idx=-1):
        for ps_body_mesh in self.meshes[frame_idx]:
            ps_body_mesh.set_transparency(transparency)
        return
    
class MotionPS:
    def __init__(self, name, mlib: motion_lib.MotionLib, char_color):

        self.name = name
        self.char_color = char_color
        self.mlib = mlib
        self.char = CharacterPS(name, char_color, mlib._kin_char_model, 1)

        # have a character frame every 0.25 seconds
        motion_length = mlib._motion_lengths[0].item()
        num_frames = int(round(4 * motion_length))
        self.sequence = MotionSequencePS(name + " motion sequence", char_color, [0.0, 0.0, 0.0], num_frames, 0.0, motion_length, self.mlib)

        self.root_offset = np.array([0.0, 0.0, 0.0])
        self.root_heading_angle = 0.0

        self.time_offset = 0.0

        self.start_retarget_time = 0.0
        self.end_retarget_time = self.mlib._motion_lengths[0].item()
        return
    
    def compute_rot_quat(self):
        z_axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor([self.root_heading_angle])
        rot_quat = torch_util.axis_angle_to_quat(z_axis, angle).squeeze(0) # this function adds a dim to the front
        return rot_quat
    
    def update_transforms(self, shadow_height = 0.0, transform_full_sequence=True):
        transform = np.eye(4)
        transform[:3, 3] = self.root_offset
        rot_quat = self.compute_rot_quat()
        rot_mat = torch_util.quat_to_matrix(rot_quat)
        transform[:3, :3] = rot_mat

        if transform_full_sequence:
            self.sequence.mesh.set_transform(transform)

        # pos offset 
        self.char.pos_offset = torch.tensor(self.root_offset, dtype=torch.float32)
        self.char.rot_offset = rot_quat#pos_offset = char_pos_offset)
        self.char.update_transforms(shadow_height)
        return
    
    def update_sequence(self, start_time, end_time, num_frames):
        self.sequence = MotionSequencePS(self.name + " motion sequence", self.char_color, [0.0, 0.0, 0.0], 
                                                 num_frames, start_time, end_time, self.mlib)
        return
    
    def apply_transforms_to_motion_data(self, frame_slice=None, vis_fps=None):
        motion_frames = MotionFrames(root_pos = self.mlib._frame_root_pos, 
                                     root_rot = self.mlib._frame_root_rot,
                                     joint_rot = self.mlib._frame_joint_rot,
                                     contacts = self.mlib._frame_contacts)
        if frame_slice is None:
            new_frames = medit_lib.rotate_motion(motion_frames, self.compute_rot_quat(), torch.tensor([0.0, 0.0, 0.0]))
            new_frames.root_pos += torch.from_numpy(self.root_offset)
        else:
            new_frames = motion_frames.get_copy()
            sliced_frames = medit_lib.rotate_motion(motion_frames.get_slice(frame_slice), self.compute_rot_quat(), torch.tensor([0.0, 0.0, 0.0]))
            sliced_frames.root_pos += torch.from_numpy(self.root_offset)
            new_frames.set_slice(sliced_frames, frame_slice)
        
        self.update_mlib_motion_frames(new_frames, vis_fps)
        return
    
    def update_mlib_motion_frames(self, new_frames: MotionFrames, vis_fps=None):
        self.mlib = motion_lib.MotionLib.from_frames(frames = new_frames, char_model=self.char.char_model,
                                                     device=self.mlib._device, fps=self.mlib._motion_fps[0].item(), 
                                                     loop_mode=motion_lib.LoopMode(self.mlib._motion_loop_modes[0].item()),
                                                     contact_info=self.mlib._contact_info)

        self.root_offset[:] = 0.0
        self.root_heading_angle = 0.0
        self.update_transforms()

        motion_length = self.mlib._motion_lengths[0].item()
        if vis_fps is None:
            num_frames = int(round(15 * motion_length))
        else:
            num_frames = int(round(vis_fps * motion_length))
        
        if num_frames < 2:
            num_frames = max(2, self.mlib._motion_num_frames[0].item())
        self.update_sequence(0.0, motion_length, num_frames)
        return
    
    def set_enabled(self, sequence_val, char_val):
        self.sequence.set_enabled(sequence_val)
        self.char.set_enabled(char_val)
        return
    
    def set_disabled(self):
        self.char.set_enabled(False)
        self.sequence.set_enabled(False)
        return
    
    def set_color(self, val):
        self.char_color = np.array(val)
        self.char.set_color(self.char_color)
        self.sequence.mesh.set_color(self.char_color)
        return
    
    def set_to_time(self, t):
        self.char.set_to_time(t, 1.0/self.mlib._motion_fps[0].item(), self.mlib)
        return