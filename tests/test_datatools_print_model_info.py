"""scripts/tools/print_model_info.py: body and DOF order of a character model."""
import subprocess
import sys
from pathlib import Path

import pytest

import parc.anim.kin_char_model as kin_char_model
from scripts.tools import print_model_info

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "scripts" / "tools" / "print_model_info.py"
G1_XML = REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
HUMANOID_XML = REPO_ROOT / "data" / "assets" / "humanoid.xml"


def _run(*args, cwd):
    return subprocess.run([sys.executable, str(TOOL)] + [str(a) for a in args], cwd=str(cwd),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def test_g1(tmp_path):
    result = _run(G1_XML, cwd=tmp_path)
    assert result.returncode == 0, result.stdout
    assert "[CONTACT BODIES ORDER] (Total: 36)" in result.stdout
    assert "Total Joint DOFs: 29" in result.stdout
    assert "35: right_wrist_yaw_link" in result.stdout


def test_default_is_the_repo_humanoid(tmp_path):
    # run from elsewhere: the default path must not depend on the working directory
    result = _run(cwd=tmp_path)
    assert result.returncode == 0, result.stdout
    assert str(HUMANOID_XML) in result.stdout
    assert "[CONTACT BODIES ORDER] (Total: 15)" in result.stdout
    assert "Total Joint DOFs: 28" in result.stdout


def test_missing_file_fails(tmp_path):
    result = _run(tmp_path / "missing.xml", cwd=tmp_path)
    assert result.returncode == 1
    assert "File not found" in result.stdout


@pytest.mark.parametrize("xml", [G1_XML, HUMANOID_XML], ids=["g1", "humanoid"])
def test_order_matches_kin_char_model(xml):
    # the printed orders are the contact columns and frames[6:] columns PARC uses
    bodies, joints = print_model_info.parse_model_structure(str(xml))
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(xml))
    assert bodies == char_model.get_body_names()

    dof_bodies = []
    for joint in joints:
        dof_bodies += [joint["body"]] * joint["dim"]
    expected = []
    for j in range(1, char_model.get_num_joints()):
        expected += [char_model.get_body_name(j)] * char_model.get_joint_dof_dim(j)
    assert dof_bodies == expected
