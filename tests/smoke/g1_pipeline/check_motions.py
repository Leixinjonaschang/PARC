"""Check that motion files written by the pipeline load and match the character.

For every .pkl given (directories are searched recursively): the MS file loads,
joint_rot is [N, num_joints - 1, 4], body_contacts is [N, num_bodies], and
MotionLib.from_file loads it. Terrain-only files (only_gen output) are reported.

    python tests/smoke/g1_pipeline/check_motions.py --char_file data/assets/unitree_g1/g1_mocap_29dof.xml <files or dirs>
"""
import argparse
import pathlib
import sys

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.file_io_helper as file_io_helper


def check_file(path, char_model):
    data = file_io_helper.load_ms_file(str(path))
    misc_keys = sorted((data.misc_data or {}).keys())
    if data.motion_data is None:
        return "terrain only, hf {}, misc {}".format(tuple(data.terrain_data.hf.shape), misc_keys)

    motion = data.motion_data
    num_frames = motion.root_pos.shape[0]
    expected_joint_rot = (num_frames, char_model.get_num_joints() - 1, 4)
    expected_contacts = (num_frames, char_model.get_num_bodies())
    assert tuple(motion.joint_rot.shape) == expected_joint_rot, (tuple(motion.joint_rot.shape), expected_joint_rot)
    assert tuple(motion.body_contacts.shape) == expected_contacts, (tuple(motion.body_contacts.shape), expected_contacts)
    hf_mask_inds = (data.misc_data or {}).get(file_io_helper.HF_MASK_INDS_KEY)
    if hf_mask_inds is not None:
        assert len(hf_mask_inds) == num_frames, (len(hf_mask_inds), num_frames)

    mlib = motion_lib.MotionLib.from_file(str(path), char_model, "cpu", contact_info=True)
    assert mlib.num_motions() == 1
    return "joint_rot {}, contacts {}, {} fps, hf {}, misc {}, MotionLib {:.2f} s".format(
        tuple(motion.joint_rot.shape), tuple(motion.body_contacts.shape), motion.fps,
        tuple(data.terrain_data.hf.shape), misc_keys, mlib.get_total_length())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--char_file", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(args.char_file)

    files = []
    for path in map(pathlib.Path, args.paths):
        files.extend(sorted(path.rglob("*.pkl")) if path.is_dir() else [path])
    if not files:
        print("no motion files found")
        return 1

    results = [(f, check_file(f, char_model)) for f in files]
    print()
    for f, result in results:
        print("OK", f, "|", result)
    print("checked {} files".format(len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
