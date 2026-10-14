"""Unitree G1 support in KinCharModel and geom_util (P1a): MJCF parsing, body-name aliases,
kinematics parity with develop, and minimal point samples.

tests/fixtures/charmodel_g1_develop_reference.npz holds develop's results for the same G1
file (written by tests/fixtures/charmodel_g1_develop_reference.py under develop's torch 1.13).
"""
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.geom_util as geom_util

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "data" / "assets" / "unitree_g1"
G1_FILE = G1_DIR / "g1_mocap_29dof.xml"
HUMANOID_FILE = REPO_ROOT / "data" / "assets" / "humanoid.xml"
DEVELOP_REFERENCE_FILE = REPO_ROOT / "tests" / "fixtures" / "charmodel_g1_develop_reference.npz"

G1_BODIES_WITHOUT_GEOMS = ["left_toe_link", "pelvis_contour_link", "right_toe_link",
                           "head_link", "head_mocap", "imu_in_torso"]


def load_char(char_file):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(char_file))
    return char_model


@pytest.fixture(scope="module")
def g1():
    return load_char(G1_FILE)


@pytest.fixture(scope="module")
def humanoid():
    return load_char(HUMANOID_FILE)


@pytest.fixture(scope="module")
def develop_ref():
    with np.load(DEVELOP_REFERENCE_FILE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def get_joint_by_name(char_model, joint_name):
    joints = [j for j in char_model._joints if j.name == joint_name]
    assert len(joints) == 1, joint_name
    return joints[0]


# ---------------------------------------------------------------- loading and limits

def test_g1_counts(g1):
    assert (g1.get_num_bodies(), g1.get_num_joints(), g1.get_dof_size()) == (36, 36, 29)
    joint_types = [j.joint_type for j in g1._joints]
    assert joint_types.count(kin_char_model.JointType.HINGE) == 29
    assert joint_types.count(kin_char_model.JointType.FIXED) == 6


def test_g1_limits_are_radians(g1):
    knee = get_joint_by_name(g1, "left_knee_joint")
    assert knee.joint_type == kin_char_model.JointType.HINGE
    np.testing.assert_allclose(knee.limits.numpy(), [-0.0873, 2.8798], atol=1e-4)
    assert g1._lower_dof_limits[knee.dof_idx].item() == knee.limits[0].item()
    assert g1._upper_dof_limits[knee.dof_idx].item() == knee.limits[1].item()

    # every hinge range read back verbatim from the MJCF (angle="radian")
    xml_ranges = {}
    for joint_elem in ET.parse(str(G1_FILE)).getroot().iter("joint"):
        if "range" in joint_elem.attrib:
            xml_ranges[joint_elem.attrib["name"]] = [float(v) for v in joint_elem.attrib["range"].split()]
    hinges = [j for j in g1._joints if j.joint_type == kin_char_model.JointType.HINGE]
    assert sorted(j.name for j in hinges) == sorted(xml_ranges)
    for joint in hinges:
        np.testing.assert_allclose(joint.limits.numpy(), xml_ranges[joint.name], rtol=1e-6, err_msg=joint.name)


@pytest.mark.parametrize("char_file", ["g1_mocap_29dof.xml", "g1_mocap_29dof_with_hands.xml"])
def test_g1_files_load_with_radian_limits(char_file):
    char_model = load_char(G1_DIR / char_file)
    lower, upper = char_model._lower_dof_limits, char_model._upper_dof_limits
    assert lower.shape == upper.shape == (char_model.get_dof_size(),)
    assert torch.all(lower < upper)
    # read as degrees, ranges would shrink 57x (the widest G1 range to ~0.074 rad)
    assert torch.max(upper - lower).item() > 1.0
    assert torch.all(lower >= -2.0 * math.pi) and torch.all(upper <= 2.0 * math.pi)


def test_g1_dof_to_rot_returns_35_joint_rotations(g1):
    dof = torch.zeros(7, g1.get_dof_size())
    joint_rot = g1.dof_to_rot(dof)
    assert joint_rot.shape == (7, 35, 4)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(7, 35, 4)
    assert torch.equal(joint_rot, identity)


def test_g1_dof_rot_dof_round_trip(g1):
    lower, upper = g1._lower_dof_limits, g1._upper_dof_limits
    gen = torch.Generator().manual_seed(7)
    dof = lower + (upper - lower) * torch.rand(256, g1.get_dof_size(), generator=gen)
    dof[0] = 0.0
    dof_rt = g1.rot_to_dof(g1.dof_to_rot(dof))
    assert torch.max(torch.abs(dof_rt - dof)).item() < 1e-5


# ---------------------------------------------------------------- parity with develop

def test_g1_structure_matches_develop(g1, develop_ref):
    assert g1.get_body_names() == develop_ref["body_names"].tolist()
    assert [g1.get_num_bodies(), g1.get_num_joints(), g1.get_dof_size()] == develop_ref["counts"].tolist()
    assert [j.name for j in g1._joints] == develop_ref["joint_names"].tolist()
    assert [j.joint_type.value for j in g1._joints] == develop_ref["joint_types"].tolist()
    assert [j.dof_idx for j in g1._joints] == develop_ref["joint_dof_idx"].tolist()


def test_g1_limits_match_develop(g1, develop_ref):
    np.testing.assert_allclose(g1._lower_dof_limits.numpy(), develop_ref["lower_dof_limits"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(g1._upper_dof_limits.numpy(), develop_ref["upper_dof_limits"], rtol=0, atol=1e-5)


def test_g1_kinematics_match_develop(g1, develop_ref):
    root_pos = torch.from_numpy(develop_ref["in_root_pos"])
    root_rot = torch.from_numpy(develop_ref["in_root_rot"])
    joint_rot = g1.dof_to_rot(torch.from_numpy(develop_ref["in_dof"]))
    body_pos, body_rot = g1.forward_kinematics(root_pos, root_rot, joint_rot)
    np.testing.assert_allclose(joint_rot.numpy(), develop_ref["joint_rot"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(body_pos.numpy(), develop_ref["fk_body_pos"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(body_rot.numpy(), develop_ref["fk_body_rot"], rtol=0, atol=1e-5)


def test_g1_minimal_point_samples_match_develop(g1, develop_ref):
    points = geom_util.get_minimal_char_point_samples(g1)
    assert [p.shape[0] for p in points] == develop_ref["min_points_counts"].tolist()
    np.testing.assert_allclose(torch.cat(points).numpy(), develop_ref["min_points"], rtol=0, atol=1e-5)


# ---------------------------------------------------------------- minimal point samples

def test_g1_minimal_point_samples(g1):
    points = geom_util.get_minimal_char_point_samples(g1)
    assert len(points) == g1.get_num_bodies() == 36
    for body_name, body_points in zip(g1.get_body_names(), points):
        assert body_points.dim() == 2 and body_points.shape[1] == 3, body_name
        assert body_points.shape[0] >= 1, body_name
        assert body_points.dtype == torch.float32 and body_points.device == g1._device, body_name
        assert torch.isfinite(body_points).all(), body_name

    geomless = [n for b, n in enumerate(g1.get_body_names()) if len(g1.get_geoms(b)) == 0]
    assert sorted(geomless) == sorted(G1_BODIES_WITHOUT_GEOMS)
    for name in geomless:
        assert torch.equal(points[g1.get_body_id(name)], torch.zeros(1, 3)), name


def test_g1_mesh_and_cylinder_point_samples(g1):
    knee_id = g1.get_body_id("left_knee_link")
    knee_geoms = g1.get_geoms(knee_id)
    assert [g._shape_type for g in knee_geoms] == [kin_char_model.GeomType.MESH] * 2
    bounds = g1._meshes[knee_geoms[0]._mesh_name].bounds
    mesh_points = geom_util.get_mesh_capsule_corner_samples(bounds, "cpu")
    major_axis = int(np.argmax(bounds[1] - bounds[0]))
    other_axes = [a for a in range(3) if a != major_axis]
    assert torch.equal(mesh_points[0, other_axes], mesh_points[1, other_axes])
    np.testing.assert_allclose((mesh_points[0, major_axis] - mesh_points[1, major_axis]).item(),
                               0.8 * (bounds[1] - bounds[0])[major_axis], rtol=1e-5)
    # identity geom quat and zero offset: each of the two mesh geoms adds those two points
    points = geom_util.get_minimal_char_point_samples(g1)[knee_id]
    assert torch.equal(points, torch.cat([mesh_points, mesh_points]))

    shoulder_id = g1.get_body_id("left_shoulder_pitch_link")
    shoulder_geoms = g1.get_geoms(shoulder_id)
    assert [g._shape_type for g in shoulder_geoms] == [kin_char_model.GeomType.MESH, kin_char_model.GeomType.CYLINDER]
    shoulder_points = geom_util.get_minimal_char_point_samples(g1)[shoulder_id]
    assert shoulder_points.shape == (3, 3)
    assert torch.equal(shoulder_points[2], shoulder_geoms[1]._offset)


# ---------------------------------------------------------------- body-name aliases

@pytest.mark.parametrize("name, expected", [
    ("left_foot", "left_ankle_roll_link"),
    ("right_foot", "right_ankle_roll_link"),
    ("left_hand", "left_wrist_yaw_link"),
    ("right_hand", "right_wrist_yaw_link"),
    ("pelvis", "pelvis"),
    ("left_ankle_roll_link", "left_ankle_roll_link"),
])
def test_g1_resolve_body_name(g1, name, expected):
    assert g1.resolve_body_name(name) == expected
    assert g1.has_body(name)
    assert g1.get_body_id(name) == g1.get_body_names().index(expected)
    assert g1.get_joint_id(name) == g1.get_body_id(name) - 1


def test_unknown_body_name_raises_with_available_names(g1):
    assert not g1.has_body("no_such_body")
    for lookup in [g1.resolve_body_name, g1.get_body_id, g1.get_joint_id]:
        with pytest.raises(AssertionError) as excinfo:
            lookup("no_such_body")
        message = str(excinfo.value)
        assert "no_such_body" in message
        assert all(name in message for name in g1.get_body_names())


def test_humanoid_names_resolve_exactly(humanoid):
    for body_id, name in enumerate(humanoid.get_body_names()):
        assert humanoid.resolve_body_name(name) == name
        assert humanoid.get_body_id(name) == body_id
    # aliases only map canonical names to other models' links, not the reverse
    assert not humanoid.has_body("left_ankle_roll_link")


@pytest.mark.parametrize("body_names, expected", [
    (["left_wrist_yaw_link", "left_rubber_hand"], "left_wrist_yaw_link"),
    (["left_rubber_hand"], "left_rubber_hand"),
    (["left_wrist_yaw_link", "left_hand"], "left_hand"),
])
def test_aliases_exact_name_first_then_in_order(tmp_path, body_names, expected):
    bodies = "".join('<body name="{}" pos="0 0 -0.1"><geom size="0.01"/>'.format(n) for n in body_names)
    xml = ('<mujoco><worldbody><body name="root"><freejoint/><geom size="0.1"/>{}{}</body></worldbody></mujoco>'
           .format(bodies, "</body>" * len(body_names)))
    path = tmp_path / "chain.xml"
    path.write_text(xml)
    char_model = load_char(path)
    assert char_model.resolve_body_name("left_hand") == expected
    assert char_model.get_body_id("left_hand") == char_model.get_body_names().index(expected)


# ---------------------------------------------------------------- MJCF parsing edge cases

def write_mjcf(tmp_path, compiler="", default="", joints='<joint name="j" axis="0 1 0" range="-90 45"/>',
               geom_type=""):
    xml = """<mujoco model="test">
  {compiler}
  {default}
  <worldbody>
    <body name="root">
      <freejoint name="root"/>
      <geom {geom_type} size="0.1"/>
      <body name="link" pos="0 0 -0.5">
        {joints}
        <geom {geom_type} size="0.05" pos="0 0 -0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".format(compiler=compiler, default=default, joints=joints, geom_type=geom_type)
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return path


@pytest.mark.parametrize("compiler", ["", '<compiler angle="degree"/>', '<compiler meshdir="meshes"/>'])
def test_degree_ranges(tmp_path, compiler):
    char_model = load_char(write_mjcf(tmp_path, compiler=compiler))
    joint = char_model.get_joint(1)
    assert joint.joint_type == kin_char_model.JointType.HINGE
    np.testing.assert_allclose(joint.limits.numpy(), [-math.pi / 2.0, math.pi / 4.0], rtol=1e-6)


def test_radian_ranges(tmp_path):
    char_model = load_char(write_mjcf(tmp_path, compiler='<compiler angle="radian"/>',
                                      joints='<joint name="j" axis="0 1 0" range="-1.5 0.25"/>'))
    np.testing.assert_allclose(char_model.get_joint(1).limits.numpy(), [-1.5, 0.25], rtol=1e-6)


@pytest.mark.parametrize("default", [
    "",
    "<default/>",
    '<default><joint type="hinge"/></default>',
    '<default><default class="body"><joint type="hinge"/></default></default>',
    '<default><default class="other"><geom type="box"/></default></default>',
])
def test_default_joint_type_falls_back_to_hinge(tmp_path, default):
    char_model = load_char(write_mjcf(tmp_path, default=default))
    assert char_model.get_joint(1).joint_type == kin_char_model.JointType.HINGE
    assert char_model.get_dof_size() == 1


def test_default_joint_type_lookup_order():
    parse = kin_char_model.KinCharModel("cpu")._parse_default_joint_type
    assert parse(ET.fromstring("<mujoco/>")) is None
    assert parse(ET.fromstring("<mujoco><default><geom type='box'/></default></mujoco>")) is None
    assert parse(ET.fromstring("<mujoco><default><joint type='slide'/></default></mujoco>")) == "slide"
    nested_first = ("<mujoco><default><joint type='slide'/>"
                    "<default class='body'><joint type='hinge'/></default></default></mujoco>")
    assert parse(ET.fromstring(nested_first)) == "hinge"


def test_untyped_three_joint_body_is_spherical(tmp_path):
    joints = "\n".join('<joint name="s_{}" axis="{}" range="-0.5 0.5"/>'.format(a, axis)
                       for a, axis in zip("xyz", ["1 0 0", "0 1 0", "0 0 1"]))
    char_model = load_char(write_mjcf(tmp_path, compiler='<compiler angle="radian"/>', joints=joints))
    joint = char_model.get_joint(1)
    assert joint.joint_type == kin_char_model.JointType.SPHERICAL and joint.name == "s"
    np.testing.assert_allclose(joint.limits.numpy(), [[-0.5, 0.5]] * 3, rtol=1e-6)


def test_unsupported_joint_type_message(tmp_path):
    path = write_mjcf(tmp_path, joints='<joint name="j" type="slide" axis="0 0 1" range="0 1"/>')
    with pytest.raises(AssertionError, match="Unsupported joint type: slide"):
        load_char(path)


@pytest.mark.parametrize("default, expected", [
    ("", kin_char_model.GeomType.SPHERE),
    ('<default><joint type="hinge"/></default>', kin_char_model.GeomType.SPHERE),
    ('<default><geom type="box"/></default>', kin_char_model.GeomType.BOX),
    ('<default><default class="body"><geom type="capsule"/></default></default>', kin_char_model.GeomType.CAPSULE),
])
def test_default_geom_type(tmp_path, default, expected):
    char_model = load_char(write_mjcf(tmp_path, default=default, geom_type='type="sphere"'))
    assert char_model._default_geom_type == expected
