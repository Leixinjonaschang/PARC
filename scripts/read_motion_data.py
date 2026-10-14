import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

"""
Example script for reading the motion data format

    python -m scripts.read_motion_data [motion file, default data/motion_terrains/sfu.pkl]
"""

import parc.util.file_io as file_io

MOTION_FILE = "data/motion_terrains/sfu.pkl"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    motion_file = argv[0] if len(argv) > 0 else MOTION_FILE

    ms_file_data = file_io.load_ms_file(motion_file)

    print(ms_file_data.motion_data.root_pos)
    print(ms_file_data.motion_data.root_rot)
    print(ms_file_data.motion_data.joint_rot)
    print(ms_file_data.motion_data.body_contacts)
    print(ms_file_data.motion_data.fps)


if __name__ == "__main__":
    main()
