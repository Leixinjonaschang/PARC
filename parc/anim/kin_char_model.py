import copy
import enum
import os
import xml.etree.ElementTree as ET
from typing import List, Union

import numpy as np
import torch
import trimesh as trimesh

import parc.util.torch_util as torch_util

_EPS = 1e-6

class JointType(enum.Enum):
    ROOT = 0
    HINGE = 1
    SPHERICAL = 2
    FIXED = 3

class Joint():
    def __init__(self, name, joint_type, axis, limits=None):
        self.name = name
        self.joint_type = joint_type
        self.axis = axis
        self.dof_idx = -1
        self.limits = limits
        return
    
    def get_copy(self, new_device):
        return Joint(self.name, 
                     self.joint_type, 
                     axis = None if self.axis is None else self.axis.clone().to(device=new_device),
                     limits = None if self.limits is None else self.limits.clone().to(device=new_device))

    def get_dof_dim(self):
        if (self.joint_type == JointType.ROOT):
            dof_dim = 0
        elif (self.joint_type == JointType.HINGE):
            dof_dim = 1
        elif (self.joint_type == JointType.SPHERICAL):
            dof_dim = 3
        elif (self.joint_type == JointType.FIXED):
            dof_dim = 0
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)
        return dof_dim

    def get_joint_dof(self, dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        j_dof = dof[..., dof_idx:dof_idx + dof_dim]
        return j_dof

    def set_joint_dof(self, j_dof, out_dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        out_dof[..., dof_idx:dof_idx + dof_dim] = j_dof
        return

    def dof_to_rot(self, dof):
        rot_shape = list(dof.shape[:-1])
        rot_shape = rot_shape + [4]
        rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        if (self.joint_type == JointType.ROOT):
            rot[..., -1] = 1
        elif (self.joint_type == JointType.HINGE):
            axis = self.axis
            axis_shape = rot[..., 0:3].shape
            axis = torch.broadcast_to(axis, axis_shape)
            dof = dof.squeeze(-1)
            rot[:] = torch_util.axis_angle_to_quat(axis, dof)
        elif (self.joint_type == JointType.SPHERICAL):
            rot[:] = torch_util.exp_map_to_quat(dof)
        elif (self.joint_type == JointType.FIXED):
            rot[..., -1] = 1
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)

        return rot

    def rot_to_dof(self, rot):
        dof_dim = self.get_dof_dim()
        dof_shape = list(rot.shape[:-1])
        dof_shape = dof_shape + [dof_dim]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)

        if (self.joint_type == JointType.ROOT):
            pass
        elif (self.joint_type == JointType.HINGE):
            j_axis = self.axis
            axis, angle = torch_util.quat_to_axis_angle(rot)
            dot_axis = torch.sum(j_axis * axis, dim=-1)
            angle[dot_axis < 0] *= -1
            dof[:] = angle.unsqueeze(-1)
        elif (self.joint_type == JointType.SPHERICAL):
            dof[:] = torch_util.quat_to_exp_map(rot)
        elif (self.joint_type == JointType.FIXED):
            pass
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)

        return dof

class GeomType(enum.Enum):
    BOX=0
    SPHERE=1
    CAPSULE=2
    CYLINDER=3
    MESH=4

class Geom:
    def __init__(self, shape_type, offset, dims, device, quat=None, radius=None, mesh_name=None, name=None):
        self._shape_type = shape_type
        if not isinstance(offset, torch.Tensor):
            self._offset = torch.tensor(offset, dtype=torch.float32, device=device) # bodies are rotated with their offset
        else:
            self._offset = offset

        if not isinstance(dims, torch.Tensor):
            self._dims = torch.tensor(dims, dtype=torch.float32, device=device)
        else:
            self._dims = dims
            
        self._radius = radius # just for capsule
        self._mesh_name = mesh_name

        self._quat = quat
        self._name = name

    def get_copy(self, new_device):
        offset = self._offset.clone().to(device=new_device)
        dims = self._dims.clone().to(device=new_device)

        if hasattr(self, "_quat"):
            quat = self._quat
        else:
            quat = None
        return Geom(shape_type=self._shape_type,
                    offset=offset,
                    dims=dims,
                    device=new_device,
                    quat=quat,
                    radius=self._radius,
                    mesh_name=self._mesh_name,
                    name=self._name)

class KinCharModel():
    def __init__(self, device):
        self._device = torch.device(device)
        return

    def init(self, body_names, parent_indices, local_translation, local_rotation, joints, geoms=None, contact_body_names=None):
        num_bodies = len(body_names)
        assert(len(parent_indices) == num_bodies)
        assert(len(local_translation) == num_bodies)
        assert(len(local_rotation) == num_bodies)
        assert(len(joints) == num_bodies)

        self._body_names = body_names
        if not isinstance(parent_indices, torch.Tensor):
            self._parent_indices = torch.tensor(parent_indices, device=self._device, dtype=torch.long)
        else:
            self._parent_indices = parent_indices

        if not isinstance(local_translation, torch.Tensor):
            self._local_translation = torch.tensor(np.array(local_translation), device=self._device, dtype=torch.float32)
        else:
            self._local_translation = local_translation
        self._original_local_translation = self._local_translation.clone()

        if not isinstance(local_rotation, torch.Tensor):
            self._local_rotation = torch.tensor(np.array(local_rotation), device=self._device, dtype=torch.float32)
        else:
            self._local_rotation = local_rotation

        self._joints = joints
        
        self._dof_size = self._label_dof_indices(self._joints)
        self._name_body_map = self._build_name_body_map(self._body_names)

        self._lower_dof_limits, self._upper_dof_limits = self._gather_joint_limits(self._joints)
        self._geoms = geoms

        if contact_body_names is None:
            contact_body_names = copy.deepcopy(body_names)
            self._contact_body_ids = torch.arange(start=0, end=self.get_num_bodies(), dtype=torch.int64, device=self._device)
        else:
            self._contact_body_ids = []
            for name in contact_body_names:
                body_id = self.get_body_id(name)
                self._contact_body_ids.append(body_id)
            self._contact_body_ids = torch.tensor(self._contact_body_ids, dtype=torch.int64, device=self._device)
        self._contact_body_names = contact_body_names
        self._name_contact_body_map = self._build_name_body_map(self._contact_body_names)
        return
    
    def get_copy(self, new_device = None):
        if new_device is None:
            new_device = self._device
        body_names = copy.deepcopy(self._body_names)
        parent_indices = self._parent_indices.clone().to(device=new_device)
        local_translation = self._local_translation.clone().to(device=new_device)
        local_rotation = self._local_rotation.clone().to(device=new_device)
        joints = []
        for joint in self._joints:
            joints.append(joint.get_copy(new_device))
        
        geoms = []
        for body_geoms in self._geoms:
            geoms.append([])
            for body_geom in body_geoms:
                geoms[-1].append(body_geom.get_copy(new_device))

        contact_body_names = copy.deepcopy(self._contact_body_names)

        meshes = None
        if hasattr(self, "_meshes") and self._meshes is not None:
            meshes = copy.deepcopy(self._meshes)

        char_model = KinCharModel(new_device)
        char_model.init(body_names=body_names,
                        contact_body_names=contact_body_names,
                        parent_indices=parent_indices,
                        local_translation=local_translation,
                        local_rotation=local_rotation,
                        joints=joints,
                        geoms=geoms)

        if meshes is not None:
            char_model._meshes = meshes
        return char_model

    def load_char_file(self, char_file):
        tree = ET.parse(char_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        assert(xml_world_body is not None)

        xml_body_root = xml_world_body.find("body")
        assert(xml_body_root is not None)

        body_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints = []
        geoms = []

        default_joint_type = self._parse_default_joint_type(xml_doc_root)

        def load_mujoco_meshes(xml_path):
            """
            Loads meshes specified in the <asset> section of a MuJoCo XML file,
            respecting the <compiler meshdir="..."/> directive.

            Args:
                xml_path (str): Path to the MuJoCo XML file.

            Returns:
                dict: Dictionary mapping mesh names to trimesh.Trimesh objects.
            """
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get directory containing the XML
            xml_dir = os.path.dirname(xml_path)

            # Extract meshdir from <compiler>
            compiler = root.find('compiler')
            mesh_dir = compiler.attrib.get('meshdir', '') if compiler is not None else ''
            mesh_dir_full = os.path.normpath(os.path.join(xml_dir, mesh_dir))

            meshes = {}

            # Find <asset> section
            asset_elem = root.find('asset')
            if asset_elem is None:
                print("No assets found in xml file")
                return meshes

            

            for mesh in asset_elem.findall('mesh'):
                name = mesh.attrib.get('name')
                file_name = mesh.attrib.get('file')
                scale = np.fromstring(mesh.attrib.get("scale", "1.0 1.0 1.0"), dtype=float, sep=" ")

                if not name or not file_name:
                    continue

                mesh_path = os.path.join(mesh_dir_full, file_name)
                mesh_path = mesh_path.replace("\\", os.sep)

                try:
                    trimesh_obj = trimesh.load(mesh_path, force='mesh')
                    trimesh_obj.apply_scale(scale)
                    meshes[name] = trimesh_obj
                    print(f"Loaded mesh '{name}' from '{mesh_path}'")
                except Exception as e:
                    print(f"Failed to load mesh '{name}' from '{mesh_path}': {e}")

            return meshes
        
        def get_default_geom_type(xml_root):
            """
            Extracts the default geom type from a MuJoCo XML root node.

            Args:
                xml_root (xml.etree.ElementTree.Element): The root element of the parsed MuJoCo XML.

            Returns:
                str: The default geom type or None if not specified.
            """
            # Find the <default> section
            shape_type = None
            default_elem = xml_root.find('default')
            if default_elem is not None:
                # Check for the first <geom> in the <default> section
                geom_elem = default_elem.find('geom')
                if geom_elem is not None:
                    shape_type = geom_elem.attrib.get('type', None)  # Default type if specified
                else:
                    second_default_elem = default_elem.find('default')
                    geom_elem = second_default_elem.find('geom')
                    if geom_elem is not None:
                        shape_type = geom_elem.attrib.get('type', None)  # Default type if specified

            if shape_type == "sphere":
                shape_type = GeomType.SPHERE
            elif shape_type == "box":
                shape_type = GeomType.BOX
            elif shape_type == "capsule":
                shape_type = GeomType.CAPSULE
            elif shape_type == "cylinder":
                shape_type = GeomType.CYLINDER
            elif shape_type == "mesh":
                shape_type = GeomType.MESH
            else:
                shape_type = GeomType.SPHERE
            
            return shape_type
        
        self._default_geom_type = get_default_geom_type(xml_doc_root)

        def get_quat(geom_node):
            # mjcf quat ordering has w first
            quat = np.fromstring(geom_node.attrib.get("quat", "1.0 0.0 0.0 0.0"), dtype=float, sep=" ")
            w = quat[0]
            xyz = quat[1:]
            quat = np.zeros_like(quat)
            quat[0:3] = xyz
            quat[3] = w
            return quat

        def get_geom(xml_node, body_name):
            curr_geoms = []

            for geom_idx, geom_node in enumerate(xml_node.findall("geom")):
                shape_type = geom_node.attrib.get("type")
                geom_name = geom_node.attrib.get("name")
                if geom_name is None or geom_name == "":
                    geom_name = f"{body_name}_geom_{geom_idx}"
                if shape_type == "sphere":
                    shape_type = GeomType.SPHERE
                elif shape_type == "box":
                    shape_type = GeomType.BOX
                elif shape_type == "capsule":
                    shape_type = GeomType.CAPSULE
                elif shape_type == "cylinder":
                    shape_type = GeomType.CYLINDER
                elif shape_type == "mesh":
                    shape_type = GeomType.MESH
                else:
                    shape_type = self._default_geom_type

                if shape_type == GeomType.SPHERE or shape_type == GeomType.BOX:
                    offset_data = geom_node.attrib.get("pos")
                    if offset_data is not None:
                        offset = np.fromstring(offset_data, dtype=float, sep=" ")
                    else:
                        offset = np.array([0.0, 0.0, 0.0])

                    size_data = geom_node.attrib.get("size")
                    if size_data is None:
                        dims = np.array(0.1, dtype=float)
                    else:
                        dims = np.fromstring(size_data, dtype=float, sep=" ")

                    quat = get_quat(geom_node)
                    geom = Geom(shape_type, offset, dims, self._device, quat=quat, name=geom_name)
                elif shape_type == GeomType.CAPSULE:
                    # store info for an oriented bounding box
                    fromto_data = geom_node.attrib.get("fromto")
                    fromto = np.fromstring(fromto_data, dtype=float, sep=" ")
                    start_point = fromto[0:3]
                    end_point = fromto[3:6]
                    offset = start_point
                    dims = end_point - start_point
                    radius = float(geom_node.attrib.get("size"))
                    quat = get_quat(geom_node)
                    #print("radius:", radius)
                    geom = Geom(shape_type, offset, dims, self._device, quat=quat, radius=radius, name=geom_name)
                elif shape_type == GeomType.CYLINDER:
                    size_data = geom_node.attrib.get("size")
                    size = np.fromstring(size_data, dtype=float, sep=" ")
                    offset_data = geom_node.attrib.get("pos")
                    if offset_data is not None:
                        offset = np.fromstring(offset_data, dtype=float, sep=" ")
                    else:
                        offset = np.array([0.0, 0.0, 0.0])
                    quat = get_quat(geom_node)
                    geom = Geom(shape_type, offset, size, self._device, quat=quat, name=geom_name)

                elif shape_type == GeomType.MESH:
                    offset = np.fromstring(geom_node.attrib.get("pos", "0.0 0.0 0.0"), dtype=float, sep=" ")
                    dims = np.ones(shape=[3], dtype=float)
                    radius = 1.0
                    mesh_name = geom_node.attrib.get("mesh")

                    quat = get_quat(geom_node)

                    geom = Geom(shape_type, offset, dims, self._device,
                                radius=radius, mesh_name=mesh_name, quat=quat, name=geom_name)
                else:
                    assert False
                curr_geoms.append(geom)
            return curr_geoms

        # recursively adding all bodies into the skel_tree
        def _add_xml_body(xml_node, parent_index, body_index, default_joint_type):
            
            body_name = xml_node.attrib.get("name")
            if body_name is None:
                assert False, "Body element is missing required 'name' attribute in character file."
            # parse the local translation into float list
            pos_data = xml_node.attrib.get("pos")
            if (pos_data is None):
                pos = np.array([0.0, 0.0, 0.0])
            else:
                pos = np.fromstring(pos_data, dtype=float, sep=" ")

            curr_geoms = get_geom(xml_node, body_name)
            geoms.append(curr_geoms)
            
            rot_data = xml_node.attrib.get("quat")
            if (rot_data is None):
                rot = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                rot = np.fromstring(rot_data, dtype=float, sep=" ")
                rot_w = rot[..., 0].copy()
                rot[..., 0:3] = rot[..., 1:]
                rot[..., -1] = rot_w

            if (body_index == 0):
                curr_joint = self._build_root_joint()
            else:
                joint_data = xml_node.findall("joint")
                curr_joint = self._parse_joint(body_name, joint_data, default_joint_type)

            body_names.append(body_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(rot)
            joints.append(curr_joint)

            curr_index = body_index
            body_index += 1
            for child_body in xml_node.findall("body"):
                body_index = _add_xml_body(child_body, curr_index, body_index, default_joint_type)

            return body_index
        
        self._meshes = load_mujoco_meshes(char_file)
        _add_xml_body(xml_body_root, -1, 0, default_joint_type)

        custom_contact_info_elem = xml_doc_root.find("custom_contact_info")
        if custom_contact_info_elem is not None:
            contact_body_names = []
            for elem in custom_contact_info_elem.findall("body"):
                contact_body_name = elem.get("name")
                assert contact_body_name in body_names
                contact_body_names.append(contact_body_name)
        else:
            contact_body_names = None
            

        self.init(body_names=body_names,
                  parent_indices=parent_indices,
                  local_translation=local_translation,
                  local_rotation=local_rotation,
                  joints=joints,
                  geoms=geoms,
                  contact_body_names=contact_body_names)
        return
    
    def get_body_names(self):
        return self._body_names

    def get_body_geom_names(self) -> List[str]:
        geom_names: List[str] = []
        for body_geoms in self._geoms:
            for geom in body_geoms:
                if getattr(geom, "_name", None) is not None:
                    geom_names.append(geom._name)
        return geom_names

    def get_num_bodies(self):
        return len(self._body_names)
    
    def get_num_contact_bodies(self):
        return len(self._contact_body_names)

    def is_contact_body(self, body_name):
        return body_name in self._contact_body_names

    def create_full_contact_tensor(self, contact_tensor: torch.Tensor) -> torch.Tensor:
        """Expands a contact tensor to include all bodies.

        Args:
            contact_tensor: Tensor with contact labels for contact bodies, with
                shape ``[..., num_contact_bodies]``.

        Returns:
            Tensor with contact labels for all bodies, with shape
            ``[..., num_bodies]``. Non-contact bodies are assigned label 0.
        """

        if contact_tensor.shape[-1] != self.get_num_contact_bodies():
            assert False, (
                "Contact tensor has incompatible shape: "
                f"expected last dim {self.get_num_contact_bodies()}, got {contact_tensor.shape[-1]}"
            )

        if self.get_num_contact_bodies() == self.get_num_bodies():
            return contact_tensor

        full_shape = list(contact_tensor.shape)
        full_shape[-1] = self.get_num_bodies()
        full_contact_tensor = torch.zeros(
            size=full_shape,
            dtype=contact_tensor.dtype,
            device=contact_tensor.device,
        )

        contact_body_ids = self.get_contact_body_ids().to(device=contact_tensor.device)
        full_contact_tensor[..., contact_body_ids] = contact_tensor
        return full_contact_tensor

    # returns the id in the space of contact bodies, which may be less than all bodies
    def get_contact_body_id(self, body_name):
        assert body_name in self._contact_body_names
        return self._name_contact_body_map[body_name]
    
    def get_contact_body_ids(self):
        return self._contact_body_ids
    
    def get_contact_body_name(self, body_id):
        return self._contact_body_names[body_id]

    def get_joint(self, j) -> Joint:
        assert(j > 0)
        return self._joints[j]

    def get_parent_id(self, j):
        return self._parent_indices[j]

    def get_dof_size(self):
        return self._dof_size

    def get_joint_dof_idx(self, j):
        dof_idx = self.get_joint(j).dof_idx
        return dof_idx

    def get_joint_dof_dim(self, j):
        dof_dim = self.get_joint(j).get_dof_dim()
        return dof_dim

    def get_num_joints(self):
        return len(self._joints)

    def get_num_non_root_joints(self):
        return len(self._joints) - 1

    def dof_to_rot(self, dof):
        num_joints = self.get_num_joints()

        rot_shape = list(dof.shape[:-1])
        rot_shape = rot_shape + [num_joints - 1, 4]
        joint_rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof = joint.get_joint_dof(dof)
            j_rot = joint.dof_to_rot(j_dof)
            joint_rot[..., j - 1, :] = j_rot

        return joint_rot

    def rot_to_dof(self, rot):
        dof_shape = list(rot.shape[:-2])
        dof_shape = dof_shape + [self._dof_size]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)
        
        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof_dim = joint.get_dof_dim()
            if (j_dof_dim > 0):
                j_rot = rot[..., j - 1, :]
                j_dof = joint.rot_to_dof(j_rot)
                joint.set_joint_dof(j_dof, dof)

        return dof

    def forward_kinematics(self, root_pos, root_rot, joint_rot):
        num_joints = self.get_num_joints()
        body_pos = [None] * num_joints
        body_rot = [None] * num_joints

        body_pos[0] = root_pos
        body_rot[0] = root_rot

        for j in range(1, num_joints):
            j_rot = joint_rot[..., j - 1, :]
            local_trans = self._local_translation[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]
            
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]

            local_trans = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot = torch.broadcast_to(local_rot, parent_rot.shape)

            world_trans = torch_util.quat_rotate(parent_rot, local_trans)

            curr_pos = parent_pos + world_trans
            curr_rot = torch_util.quat_mul(local_rot, j_rot)
            curr_rot = torch_util.quat_mul(parent_rot, curr_rot)

            body_pos[j] = curr_pos
            body_rot[j] = curr_rot

        body_pos = torch.stack(body_pos, dim=-2)
        body_rot = torch.stack(body_rot, dim=-2)
        
        return body_pos, body_rot
    
    def compute_frame_dof_vel(self, joint_rot, dt):
        joint_rot0 = joint_rot[..., :-1, :, :]
        joint_rot1 = joint_rot[..., 1:, :, :]
        dof_vel = self.compute_dof_vel(joint_rot0, joint_rot1, dt)
        
        # adding an extra frame so dof_vel is same num frames as dof_pos
        final_vels = dof_vel[...,-1:, :]
        dof_vel = torch.cat([dof_vel, final_vels], dim=-2)
        return dof_vel

    def compute_dof_vel(self, joint_rot0, joint_rot1, dt):
        dof_size = self.get_dof_size()
        dof_shape = list(joint_rot0.shape[:-2])
        dof_shape = dof_shape + [dof_size]
        dof_vel = torch.zeros(dof_shape, device=joint_rot0.device, dtype=joint_rot0.dtype)

        drot = torch_util.quat_mul(torch_util.quat_conjugate(joint_rot0), joint_rot1)
        drot = torch_util.quat_normalize(drot)

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_drot = drot[..., j - 1, :]

            if (joint.joint_type == JointType.ROOT):
                pass
            elif (joint.joint_type == JointType.HINGE):
                j_axis = joint.axis
                j_dof_vel = torch_util.quat_to_exp_map(j_drot) / dt
                j_dof_vel = torch.sum(j_axis * j_dof_vel, dim=-1, keepdim=True)
                joint.set_joint_dof(j_dof_vel, dof_vel)
            elif (joint.joint_type == JointType.SPHERICAL):
                j_dof_vel = torch_util.quat_to_exp_map(j_drot) / dt
                joint.set_joint_dof(j_dof_vel, dof_vel)
            elif (joint.joint_type == JointType.FIXED):
                pass
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

        return dof_vel
    
    def get_body_name(self, body_id):
        return self._body_names[body_id]

    def get_body_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name]

    def get_joint_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name] -1 # joint arrays exclude the root

    def _resolve_body_id(self, body: Union[int, str]) -> int:
        if isinstance(body, str):
            return self.get_body_id(body)
        return int(body)

    def get_bone_length(self, body: Union[int, str]) -> float:
        """Returns the distance from the parent joint to ``body`` in meters."""
        body_id = self._resolve_body_id(body)
        bone_vec = self._local_translation[body_id]
        return torch.linalg.norm(bone_vec).item()

    def scale_bone_length(self,
                          body: Union[int, str],
                          scale: float,
                          update_original: bool = True,
                          update_geoms: bool = True) -> None:
        """Scales the bone length for ``body`` by ``scale``.

        Args:
            body: Body index or name.
            scale: Multiplicative factor applied to the bone length.
            update_original: When True, ``_original_local_translation`` is updated
                so future calls that rely on the stored baseline use the new
                length.
            update_geoms: When True, geometry offsets and extents attached to the
                parent body are stretched to match the new bone length.
        """
        if scale <= 0:
            raise ValueError("Scale must be positive")

        current_length = self.get_bone_length(body)
        self.set_bone_length(body,
                             current_length * scale,
                             update_original=update_original,
                             update_geoms=update_geoms)

    def set_bone_length(self,
                        body: Union[int, str],
                        new_length: float,
                        update_original: bool = True,
                        update_geoms: bool = True) -> None:
        """Sets the bone length for ``body`` to ``new_length`` meters."""
        if new_length <= 0:
            raise ValueError("new_length must be positive")

        body_id = self._resolve_body_id(body)
        parent_id = self._parent_indices[body_id].item()
        if parent_id < 0:
            raise ValueError("Cannot change length of the root body")

        old_vec = self._local_translation[body_id]
        old_length = torch.linalg.norm(old_vec).item()
        if old_length < _EPS:
            raise ValueError("Cannot resize a bone with zero length")

        direction = old_vec / old_length
        scale = new_length / old_length
        new_vec = direction * new_length

        self._local_translation[body_id] = new_vec
        if update_original:
            self._original_local_translation[body_id] = new_vec.clone()

        if update_geoms:
            self._update_geoms_for_bone(parent_id, direction, scale)

    def _update_geoms_for_bone(self,
                               parent_id: int,
                               direction: torch.Tensor,
                               scale: float) -> None:
        if self._geoms is None or parent_id < 0 or abs(scale - 1.0) < _EPS:
            return

        dir_norm = torch.linalg.norm(direction)
        if dir_norm < _EPS:
            return

        direction = direction / dir_norm
        parent_geoms = self._geoms[parent_id]

        for geom in parent_geoms:
            offset = getattr(geom, "_offset", None)
            if isinstance(offset, torch.Tensor) and offset.numel() == 3:
                parallel = torch.dot(offset, direction)
                if torch.abs(parallel) > _EPS:
                    parallel_vec = parallel * direction
                    perp_vec = offset - parallel_vec
                    geom._offset = perp_vec + parallel_vec * scale

            if geom._shape_type == GeomType.CAPSULE:
                dims = getattr(geom, "_dims", None)
                if isinstance(dims, torch.Tensor) and dims.numel() == 3:
                    parallel = torch.dot(dims, direction)
                    if torch.abs(parallel) > _EPS:
                        parallel_vec = parallel * direction
                        perp_vec = dims - parallel_vec
                        geom._dims = perp_vec + parallel_vec * scale
            elif geom._shape_type == GeomType.CYLINDER:
                dims = getattr(geom, "_dims", None)
                if isinstance(dims, torch.Tensor) and dims.numel() >= 2:
                    new_dims = dims.clone()
                    new_dims[1] = new_dims[1] * scale
                    geom._dims = new_dims
    
    def _build_name_body_map(self, body_names):
        name_body_map = dict()

        for body_id, body_name in enumerate(body_names):
            name_body_map[body_name] = body_id

        return name_body_map
    
    def _build_root_joint(self):
        joint = Joint(name="root",
                      joint_type=JointType.ROOT,
                      axis=None)
        return joint
    
    def _parse_joint(self, body_name, xml_joint_data, default_joint_type):
        num_joints = len(xml_joint_data)

        if (num_joints == 0):
            joint = self._parse_fixed_joint(body_name)
        elif (num_joints == 3):
            joint = self._parse_sphere_joint(xml_joint_data, default_joint_type)
        elif (num_joints == 1):
            joint_type_str = xml_joint_data[0].attrib.get("type")
            if (joint_type_str is None):
                joint_type_str = default_joint_type

            if (joint_type_str == "hinge"):
                joint = self._parse_hinge_joint(xml_joint_data[0])
            elif (joint_type_str == "fixed"):
                joint = self._parse_fixed_joint(body_name)
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint_type_str)
        else:
            assert(False), "Series joints are not supported."
        
        return joint

    def _parse_hinge_joint(self, xml_joint_data):
        joint_name = xml_joint_data.attrib.get("name")

        joint_pos_data = xml_joint_data.attrib.get("pos")
        if (joint_pos_data is not None):
            joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
            if (np.any(joint_pos)):
                assert(False), "Joint offsets are not supported"

        joint_axis = np.fromstring(xml_joint_data.attrib.get("axis"), dtype=float, sep=" ")
        joint_axis = torch.tensor(joint_axis, device=self._device, dtype=torch.float32)

        joint_limits_str = xml_joint_data.attrib.get("range")
        if joint_limits_str is None:
            assert(False), "Need joint limits"
        joint_limits = np.fromstring(joint_limits_str, dtype=float, sep=" ")
        joint_limits = torch.from_numpy(joint_limits).to(dtype=torch.float32, device=self._device)
        joint_limits *= torch.pi / 180.0

        joint = Joint(name=joint_name,
                      joint_type=JointType.HINGE,
                      axis=joint_axis,
                      limits=joint_limits)
        return joint

    def _parse_sphere_joint(self, xml_joint_data, default_joint_type):
        # consolidate series of three hinge joints into a single spherical joint
        num_joints = len(xml_joint_data)
        assert(num_joints == 3)

        is_spherical = True
        joint_limits = []
        for joint_data in xml_joint_data:
            joint_type_str = joint_data.attrib.get("type")
            if (joint_type_str is None):
                joint_type_str = default_joint_type

            joint_pos_data = joint_data.attrib.get("pos")
            if (joint_pos_data is not None):
                joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
                if (np.any(joint_pos)):
                    assert(False), "Joint offsets are not supported"

            joint_limit_str = joint_data.attrib.get("range")
            if joint_limit_str is not None:
                joint_limits.append(np.fromstring(joint_limit_str, dtype=float, sep=" "))
            else:
                assert(False), "Need joint limits"

            if (joint_type_str != "hinge"):
                is_spherical = False
                break

        joint_limits = np.stack(joint_limits)
        joint_limits = torch.from_numpy(joint_limits).to(dtype=torch.float32, device=self._device)
        joint_limits *= torch.pi / 180.0
        if (is_spherical):
            joint_name = xml_joint_data[0].attrib.get("name")
            joint_name = joint_name[:joint_name.rfind('_')]
            joint = Joint(name=joint_name,
                          joint_type=JointType.SPHERICAL,
                          axis=None,
                          limits=joint_limits)
        else:
            assert(False), "Invalid format for a spherical joint"

        return joint

    def _parse_fixed_joint(self, body_name):
        joint = Joint(name=body_name,
                      joint_type=JointType.FIXED,
                      axis=None)
        return joint
    
    def _gather_joint_limits(self, joints):
        lower_dof_limits = []
        upper_dof_limits = []
        for curr_joint in joints:
            if curr_joint.limits is None:
                continue
            if curr_joint.joint_type == JointType.HINGE:
                lower_dof_limits.append(curr_joint.limits[0].unsqueeze(0))
                upper_dof_limits.append(curr_joint.limits[1].unsqueeze(0))
            elif curr_joint.joint_type == JointType.SPHERICAL:
                lower_dof_limits.append(curr_joint.limits[:, 0])
                upper_dof_limits.append(curr_joint.limits[:, 1])
        if len(lower_dof_limits) > 0:
            lower_dof_limits = torch.cat(lower_dof_limits)
        if len(upper_dof_limits) > 0:
            upper_dof_limits = torch.cat(upper_dof_limits)
        return lower_dof_limits, upper_dof_limits

    def _label_dof_indices(self, joints):
        dof_idx = 0
        for curr_joint in joints:
            if (curr_joint is not None):
                dof_dim = curr_joint.get_dof_dim()
                curr_joint.dof_idx = dof_idx
                dof_idx += dof_dim
        return dof_idx

    def _parse_default_joint_type(self, xml_node):
        default_data = xml_node.find("default")
        default_data = default_data.findall("default")

        joint_type_str = None
        for data in default_data:
            class_data = data.attrib.get("class")
            if (class_data == "body"):
                joint_data = data.find("joint")
                if (joint_data is not None):
                    joint_type_str = joint_data.attrib.get("type")
                    break

        return joint_type_str

    def _build_body_children_map(self):
        num_joints = self.get_num_joints()
        body_children = [[] for _ in range(num_joints)]
        for j in range(num_joints):
            parent_idx = self._parent_indices[j].item()
            if (parent_idx != -1):
                body_children[parent_idx].append(j)
        return body_children


    
    def output_xml(self, output_file):
        xml_template = """<mujoco model="character">
    <statistic extent="2" center="0 0 1"/>

    <default>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
        <default class="body">
            <geom type="sphere" condim="1" friction="1.0 0.05 0.05" solimp=".9 .99 .003" solref=".015 1"/>
            <joint type="hinge" damping="0.1" stiffness="5" armature=".007" limited="true" solimplimit="0 .99 .01"/>
        </default>
    </default>

    <worldbody>
        <geom name="floor" type="plane" conaffinity="1" size="100 100 .2" material="grid"/>
{:s}
    </worldbody>

    <actuator>{:s}
    </actuator>
</mujoco>"""

        bodies_xml = self._build_bodies_xml()
        actuator_xml = self._build_actuators_xml()

        char_xml = xml_template.format(bodies_xml, actuator_xml)

        with open(output_file, "w") as out_file:
            out_file.write(char_xml)

        return

    def _build_bodies_xml(self):
        body_children = self._build_body_children_map()
        bodies_xml = self._build_body_xml(body_children=body_children, body_id=0)
        return bodies_xml

    def _build_body_xml(self, body_children, body_id):
        body_template = '''
        <body name="{:s}" pos="{:.4f} {:.4f} {:.4f}" rot="{:.4f} {:.4f} {:.4f} {:.4f}">{:s}{:s}{:s}
        </body>'''

        root_template = '''
        <body name="{:s}" pos="0 0 0" childclass="body">{:s}{:s}{:s}
        </body>'''
        
        body_name = self._body_names[body_id]
        pos = self._local_translation[body_id].cpu().numpy()
        rot = self._local_rotation[body_id].cpu().numpy()

        children_xml = ""
        children = body_children[body_id]
        if (len(children) > 0):
            for child_id in children:
                child_xml = self._build_body_xml(body_children=body_children, body_id=child_id)

                child_xml_lines = child_xml.splitlines()
                indented_xml_lines = ["\t" + l for l in child_xml_lines]
                child_xml = "\n".join(indented_xml_lines)

                children_xml += "\n" + child_xml

        joint_xml = self._build_joint_xml(body_children, body_id)
        geom_xml = self._build_geom_xml(body_children, body_id)

        is_root = body_id == 0
        if (is_root):
            body_xml = root_template.format(body_name,
                                            joint_xml,
                                            geom_xml,
                                            children_xml)
        else:
            body_xml = body_template.format(body_name,
                                            pos[0], pos[1], pos[2], 
                                            rot[3], rot[0], rot[1], rot[2], 
                                            joint_xml,
                                            geom_xml,
                                            children_xml)
        return body_xml

    def _build_joint_xml(self, body_children, body_id):
        root_template = '''
			<freejoint name="{:s}"/>'''

        joint_template = '''
            <joint name="{:s}_x" type="hinge" axis="1 0 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_y" type="hinge" axis="0 1 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_z" type="hinge" axis="0 0 1" range="-180 180" stiffness="100" damping="10" armature=".01"/>'''
            
        body_name = self._body_names[body_id]
        
        is_root = body_id == 0
        if (is_root):
            joint_xml = root_template.format(body_name)
        else:
            joint = self.get_joint(body_id)
            joint_type = joint.joint_type

            if (joint_type == JointType.HINGE):
                j_axis = joint.axis
                joint_template = '''
            <joint name="{:s}" type="hinge" axis="{:.4f} {:.4f} {:.4f}" range="-180 180" stiffness="100" damping="10" armature=".01"/>'''
                joint_xml = joint_template.format(body_name, j_axis[0], j_axis[1], j_axis[2])
            elif (joint_type == JointType.SPHERICAL):
                joint_template = '''
            <joint name="{:s}_x" type="hinge" axis="1 0 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_y" type="hinge" axis="0 1 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_z" type="hinge" axis="0 0 1" range="-180 180" stiffness="100" damping="10" armature=".01"/>'''
                joint_xml = joint_template.format(body_name, body_name, body_name)
            elif (joint_type == JointType.FIXED):
                joint_xml = ""
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

        return joint_xml

    def _build_geom_xml(self, body_children, body_id):
        joint_radius = 0.02
        bone_radius = 0.01
        joint_template = '''
            <geom type="sphere" name="{:s}" pos="0 0 0" size="{:.4f}" density="1000"/>'''
        bone_template = '''
            <geom type="capsule" name="{:s}" fromto="0 0 0 {:.4f} {:.4f} {:.4f}" size="{:.4f}" density="1000"/>'''

        body_name = self._body_names[body_id]

        geom_xml = ""

        joint_pos = self._local_translation[body_id]
        bone_len = np.linalg.norm(joint_pos)
        if (bone_len > 0):
            clamp_joint_radius = min(0.25 * bone_len, joint_radius)
            geom_xml += joint_template.format(body_name, clamp_joint_radius)

        children = body_children[body_id]
        joint_rot = self._local_rotation[body_id]
        for c in children:
            child_pos = self._local_translation[c]
            bone_pos = torch_util.quat_rotate(joint_rot, child_pos)

            child_bone_len = np.linalg.norm(bone_pos)
            if (child_bone_len > 0):
                geom_len = max(0.001, child_bone_len)
                geom_pos = geom_len * bone_pos / child_bone_len
                child_bone_radius = min(0.15 * child_bone_len, bone_radius)

                bone_xml = bone_template.format(body_name, geom_pos[0], geom_pos[1], geom_pos[2], child_bone_radius)
                geom_xml += bone_xml

        return geom_xml

    def _build_actuators_xml(self):
        motor_template = '''
        <motor name='{:s}'       	gear='100' 	joint='{:s}'/>'''

        actuators_xml = ""

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            joint_type = joint.joint_type
            body_name = self._body_names[j]

            if (joint_type == JointType.HINGE):
                actuator_xml = motor_template.format(body_name, body_name)
            elif (joint_type == JointType.SPHERICAL):
                actuator_xml = motor_template.format(body_name + "_x", body_name + "_x")
                actuator_xml += motor_template.format(body_name + "_y", body_name + "_y")
                actuator_xml += motor_template.format(body_name + "_z", body_name + "_z")
            elif (joint_type == JointType.FIXED):
                actuator_xml = ""
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

            actuators_xml += actuator_xml

        return actuators_xml
    
    def get_geoms(self, body_id) -> List[Geom]: 
        return self._geoms[body_id]
    
    def construct_frame_data(self, root_pos, root_rot_quat, joint_rot):
        root_rot = torch_util.quat_to_exp_map(root_rot_quat)
        joint_dof = self.rot_to_dof(joint_rot)
        motion_frames = torch.cat([root_pos, root_rot, joint_dof], dim=-1)
        return motion_frames
    
    def apply_joint_dof_limits(self, joint_dofs: torch.Tensor):
        num_dims = len(joint_dofs.shape)
        if num_dims == 1:
            ret_joint_dofs = torch.clamp(joint_dofs, min=self._lower_dof_limits, max=self._upper_dof_limits)
        elif num_dims == 2:
            ret_joint_dofs = torch.clamp(joint_dofs, 
                                          min=self._lower_dof_limits.unsqueeze(0), 
                                          max=self._upper_dof_limits.unsqueeze(0))
        else:
            assert False, "unsupported"
        return ret_joint_dofs
    
    def find_lowest_point(self, body_pos, body_rot):
        device = body_pos.device
        dtype = body_pos.dtype
        batch_shape = body_pos.shape[:-2]
        num_bodies = body_pos.shape[-2]

        flat_pos = body_pos.reshape(-1, num_bodies, 3)
        flat_rot = body_rot.reshape(-1, num_bodies, 4)
        batch_size = flat_pos.shape[0]

        down_dir = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
        min_point = torch.full((batch_size, 3), float("inf"), device=device, dtype=dtype)
        min_z = torch.full((batch_size,), float("inf"), device=device, dtype=dtype)

        if not hasattr(self, "_mesh_bounds_cache"):
            self._mesh_bounds_cache = {}

        for b in range(num_bodies):
            pos_b = flat_pos[:, b]
            rot_b = flat_rot[:, b]
            geoms = self._geoms[b]

            for geom in geoms:
                geom_offset = geom._offset.to(device=device, dtype=dtype)
                if geom_offset.dim() == 1:
                    geom_offset = geom_offset.unsqueeze(0).expand(batch_size, -1)

                geom_quat = getattr(geom, "_quat", None)
                if geom_quat is None:
                    geom_rot = rot_b
                else:
                    if not isinstance(geom_quat, torch.Tensor):
                        geom_quat = torch.tensor(geom_quat, device=device, dtype=dtype)
                    else:
                        geom_quat = geom_quat.to(device=device, dtype=dtype)
                    if geom_quat.dim() == 1:
                        geom_quat = geom_quat.unsqueeze(0).expand(batch_size, -1)
                    geom_rot = torch_util.quat_mul(rot_b, geom_quat)

                offset_world = torch_util.quat_rotate(rot_b, geom_offset)

                if geom._shape_type == GeomType.SPHERE:
                    radius = geom._dims.to(device=device, dtype=dtype)
                    if radius.dim() > 0:
                        radius = radius[0]
                    center = pos_b + offset_world
                    candidate = center + down_dir * radius

                elif geom._shape_type == GeomType.BOX:
                    dims = geom._dims.to(device=device, dtype=dtype)
                    if dims.dim() == 1:
                        dims = dims.unsqueeze(0).expand(batch_size, -1)
                    center = pos_b + offset_world

                    rot_mat = torch_util.quat_to_matrix(geom_rot)
                    sign = torch.sign(rot_mat[:, 2, :])
                    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                    extents = -sign * dims
                    support_offset = torch.bmm(rot_mat, extents.unsqueeze(-1)).squeeze(-1)
                    candidate = center + support_offset

                elif geom._shape_type == GeomType.CAPSULE:
                    start = pos_b + offset_world
                    end_offset = geom._offset.to(device=device, dtype=dtype) + geom._dims.to(device=device, dtype=dtype)
                    if end_offset.dim() == 1:
                        end_offset = end_offset.unsqueeze(0).expand(batch_size, -1)
                    end = pos_b + torch_util.quat_rotate(rot_b, end_offset)
                    axis_vec = end - start
                    length = torch.norm(axis_vec, dim=-1, keepdim=True)
                    axis_unit = axis_vec / torch.clamp(length, min=1e-8)
                    center = 0.5 * (start + end)
                    half_len = 0.5 * length
                    dot_ad = torch.sum(axis_unit * down_dir, dim=-1, keepdim=True)
                    sign = torch.where(dot_ad >= 0, torch.ones_like(dot_ad), -torch.ones_like(dot_ad))
                    axis_term = axis_unit * half_len * sign
                    radius = torch.tensor(geom._radius, device=device, dtype=dtype)
                    candidate = center + axis_term + down_dir * radius

                elif geom._shape_type == GeomType.CYLINDER:
                    dims = geom._dims.to(device=device, dtype=dtype)
                    radius = dims[0]
                    half_len = dims[1]
                    center = pos_b + offset_world

                    rot_mat = torch_util.quat_to_matrix(geom_rot)
                    axis_dir = rot_mat[:, :, 2]
                    dot_ad = torch.sum(axis_dir * down_dir, dim=-1, keepdim=True)
                    sign = torch.where(dot_ad >= 0, torch.ones_like(dot_ad), -torch.ones_like(dot_ad))
                    axis_term = axis_dir * (half_len * sign)
                    d_perp = down_dir.unsqueeze(0) - dot_ad * axis_dir
                    d_perp_norm = torch.norm(d_perp, dim=-1, keepdim=True)
                    radial = torch.where(
                        d_perp_norm > 1e-8,
                        radius * d_perp / d_perp_norm,
                        torch.zeros_like(d_perp),
                    )
                    candidate = center + axis_term + radial

                elif geom._shape_type == GeomType.MESH:
                    mesh_name = geom._mesh_name
                    if mesh_name in self._mesh_bounds_cache:
                        mesh_center_local, mesh_half_extents = self._mesh_bounds_cache[mesh_name]
                    else:
                        mesh_center_local = torch.zeros(3, device=device, dtype=dtype)
                        mesh_half_extents = torch.ones(3, device=device, dtype=dtype)
                        if hasattr(self, "_meshes") and self._meshes is not None and mesh_name in self._meshes:
                            bounds = self._meshes[mesh_name].bounds
                            center_local_np = (bounds[0] + bounds[1]) * 0.5
                            extents_np = (bounds[1] - bounds[0]) * 0.5
                            mesh_center_local = torch.tensor(center_local_np, device=device, dtype=dtype)
                            mesh_half_extents = torch.tensor(extents_np, device=device, dtype=dtype)
                        self._mesh_bounds_cache[mesh_name] = (mesh_center_local, mesh_half_extents)

                    mesh_center_local = mesh_center_local.to(device=device, dtype=dtype)
                    mesh_half_extents = mesh_half_extents.to(device=device, dtype=dtype)
                    center_local = mesh_center_local.unsqueeze(0).expand(batch_size, -1)
                    center = pos_b + offset_world + torch_util.quat_rotate(geom_rot, center_local)

                    rot_mat = torch_util.quat_to_matrix(geom_rot)
                    sign = torch.sign(rot_mat[:, 2, :])
                    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                    extents = -sign * mesh_half_extents.unsqueeze(0).expand(batch_size, -1)
                    support_offset = torch.bmm(rot_mat, extents.unsqueeze(-1)).squeeze(-1)
                    candidate = center + support_offset

                else:
                    continue

                cand_z = candidate[:, 2]
                mask = cand_z < min_z
                min_z = torch.where(mask, cand_z, min_z)
                min_point = torch.where(mask.unsqueeze(-1), candidate, min_point)

        min_point = min_point.reshape(batch_shape + (3,))
        return min_point
