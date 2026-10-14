"""motion_opt on the G1 and the humanoid (CPU, synthetic motions): body constraint ids via the
central name aliases, the D16 warning for bodies whose constraints are skipped, and tiny
motion_contact_optimization runs."""
import os
import sys
import warnings

import pytest
import torch

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.motion_synthesis.motion_opt.motion_optimization as moopt  # noqa: E402
import parc.util.geom_util as geom_util  # noqa: E402

# G1 bodies the generic names resolve to (KinCharModel.BODY_NAME_ALIASES)
G1_CONSTRAINT_BODIES = {6: "left_ankle_roll_link", 14: "right_ankle_roll_link",
                        28: "left_wrist_yaw_link", 35: "right_wrist_yaw_link"}
HUMANOID_CONSTRAINT_BODIES = {14: "left_foot", 11: "right_foot", 8: "left_hand", 5: "right_hand"}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def chars():
    return {"humanoid": synth.load_char("humanoid"), "g1": synth.load_char("g1")}


def _constraints(char_model, frames, terrain):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        constraints = moopt.compute_approx_body_constraints(
            root_pos=frames.root_pos, root_rot=frames.root_rot, joint_rot=frames.joint_rot,
            contacts=frames.contacts, char_model=char_model, terrain=terrain)
    return constraints, [str(w.message) for w in caught]


def _optimize(char_model, frames, terrain, body_constraints=None, num_iters=5, **kwargs):
    return moopt.motion_contact_optimization(
        src_motion_frames=frames, body_points=geom_util.get_char_point_samples(char_model),
        terrain=terrain, char_model=char_model, num_iters=num_iters, step_size=1e-2,
        w_root_pos=1.0, w_root_rot=10.0, w_joint_rot=1.0, w_smoothness=10.0, w_penetration=100.0,
        w_contact=100.0, w_sliding=10.0, w_body_constraints=500.0, w_jerk=100.0,
        body_constraints=body_constraints, max_jerk=1000.0, exp_name=None, use_wandb=False,
        log_file=None, **kwargs)


@pytest.mark.parametrize("char,expected", [("g1", G1_CONSTRAINT_BODIES), ("humanoid", HUMANOID_CONSTRAINT_BODIES)])
def test_compute_approx_body_constraints_body_ids(chars, char, expected):
    char_model = chars[char]
    frames, terrain = synth.make_motion(char_model)
    constraints, messages = _constraints(char_model, frames, terrain)

    assert len(constraints) == char_model.get_num_bodies()
    with_constraints = {b for b in range(len(constraints)) if len(constraints[b]) > 0}
    assert with_constraints == set(expected)
    for body_id, name in expected.items():
        assert char_model.get_body_name(body_id) == name

    # the hands touch the box in frames [12, 18) of the synthetic motion
    for hand in ("left_hand", "right_hand"):
        hand_constraints = constraints[char_model.get_body_id(hand)]
        assert [(c.start_frame_idx, c.end_frame_idx) for c in hand_constraints] == [(12, 17)]
        assert hand_constraints[0].constraint_point.shape == (3,)

    unsupported = [m for m in messages if "D16" in m]
    if char == "g1":
        # D16: the G1's feet and hands start with a mesh geom, so the optimizer ignores these
        assert len(unsupported) == 1
        for name in G1_CONSTRAINT_BODIES.values():
            assert "{} (MESH)".format(name) in unsupported[0]
    else:
        assert unsupported == []


def test_compute_approx_body_constraints_maps_contact_body_columns(tmp_path):
    # contacts come per contact body (MS files, MotionLib); constraints are indexed by body
    xml = open(synth.CHAR_FILES["humanoid"]).read()
    contact_bodies = ["pelvis", "right_hand", "left_hand", "right_foot", "left_foot"]
    xml = xml.replace("</mujoco>", "<custom_contact_info>{}</custom_contact_info>\n</mujoco>".format(
        "".join('<body name="{}"/>'.format(n) for n in contact_bodies)))
    char_file = tmp_path / "humanoid_5_contacts.xml"
    char_file.write_text(xml)
    char_model = synth.load_char(str(char_file))
    assert char_model.get_num_contact_bodies() == 5 < char_model.get_num_bodies()

    frames, terrain = synth.make_motion(char_model)
    assert frames.contacts.shape[-1] == 5
    constraints, _ = _constraints(char_model, frames, terrain)
    with_constraints = {b for b in range(len(constraints)) if len(constraints[b]) > 0}
    assert with_constraints == set(HUMANOID_CONSTRAINT_BODIES)


@pytest.mark.parametrize("char", ["humanoid", "g1"])
def test_motion_contact_optimization_tiny_run(chars, char):
    char_model = chars[char]
    frames, terrain = synth.make_motion(char_model, num_frames=16)
    constraints, _ = _constraints(char_model, frames, terrain)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt_frames = _optimize(char_model, frames, terrain, body_constraints=constraints)
    d16 = [str(w.message) for w in caught if "D16" in str(w.message)]
    assert len(d16) == (1 if char == "g1" else 0)

    for key in ("root_pos", "root_rot", "joint_rot", "contacts"):
        out, src = getattr(opt_frames, key), getattr(frames, key)
        assert out.shape == src.shape, key
        assert torch.isfinite(out).all(), key
    assert torch.equal(opt_frames.contacts, frames.contacts)
    assert not torch.equal(opt_frames.root_pos, frames.root_pos)  # the optimizer moved something
    assert torch.allclose(opt_frames.root_rot.norm(dim=-1), torch.ones(16), atol=1e-5)


def test_body_constraints_have_no_effect_on_g1(chars):
    # D16, documented behaviour: same result with and without the approximate constraints
    char_model = chars["g1"]
    frames, terrain = synth.make_motion(char_model, num_frames=16)
    constraints, _ = _constraints(char_model, frames, terrain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_constraints = _optimize(char_model, frames, terrain, body_constraints=constraints)
    without = _optimize(char_model, frames, terrain, body_constraints=None)
    assert torch.equal(with_constraints.root_pos, without.root_pos)
    assert torch.equal(with_constraints.joint_rot, without.joint_rot)


def test_constraints_on_bodies_without_geoms_are_skipped(chars):
    # left_toe_link has no geom; upstream indexed get_geoms(b)[0] and raised IndexError
    char_model = chars["g1"]
    toe_id = char_model.get_body_id("left_toe_link")
    assert len(char_model.get_geoms(toe_id)) == 0
    frames, terrain = synth.make_motion(char_model, num_frames=12)
    constraints = [[] for _ in range(char_model.get_num_bodies())]
    constraints[toe_id] = [moopt.BodyConstraint(start_frame_idx=0, end_frame_idx=3,
                                                constraint_point=torch.tensor([0.0, 0.0, 0.0]))]
    with pytest.warns(UserWarning, match=r"left_toe_link \(no geom\)"):
        opt_frames = _optimize(char_model, frames, terrain, body_constraints=constraints, num_iters=2)
    assert torch.isfinite(opt_frames.root_pos).all()


def test_yield_intermediate_matches_plain_run(chars):
    # used by the MotionScope optimization GUI to animate the optimization
    char_model = chars["humanoid"]
    frames, terrain = synth.make_motion(char_model, num_frames=12)
    plain = _optimize(char_model, frames, terrain, num_iters=4)

    steps = _optimize(char_model, frames, terrain, num_iters=4, yield_intermediate=True)
    intermediate = []
    while True:
        try:
            intermediate.append(next(steps))
        except StopIteration as stop:
            final = stop.value
            break
    assert len(intermediate) == 4
    assert all(isinstance(f, type(plain)) for f in intermediate)
    for key in ("root_pos", "root_rot", "joint_rot", "contacts"):
        assert torch.equal(getattr(final, key), getattr(plain, key)), key
        assert torch.equal(getattr(intermediate[-1], key), getattr(plain, key)), key
