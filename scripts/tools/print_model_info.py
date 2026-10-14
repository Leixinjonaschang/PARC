"""Print a MuJoCo character model's body order (contact columns) and joint DOF order (frames[6:]).

Standard library only.  Usage, from anywhere:

    python scripts/tools/print_model_info.py [<character .xml>]

The default is data/assets/humanoid.xml in this repository.  The G1
(data/assets/unitree_g1/g1_mocap_29dof.xml) has 36 bodies and 29 DOFs.
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_XML = Path(__file__).resolve().parents[2] / "data" / "assets" / "humanoid.xml"


def parse_model_structure(xml_path):
    """Returns (body names, joints) in depth-first order; joints are dicts with name, type, dim, body."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    bodies = []
    joints = []

    # Recursive function to parse bodies and joints
    def parse_body(body_elem, parent_name="world"):
        body_name = body_elem.attrib.get('name', 'unnamed')
        bodies.append(body_name)

        # Find joints in this body (<freejoint> is not a <joint> and is skipped like type="free")
        for joint in body_elem.findall('joint'):
            joint_name = joint.attrib.get('name', 'unnamed')
            joint_type = joint.attrib.get('type', 'hinge') # default to hinge if not specified

            # Determine dim based on type
            dim = 1
            if joint_type == 'free':
                dim = 6 # usually root, skipped in joint dof array
            elif joint_type == 'ball' or joint_type == 'spherical':
                dim = 3
            # hinge and slide are 1

            if joint_type != 'free':
                joints.append({'name': joint_name, 'type': joint_type, 'dim': dim, 'body': body_name})

        # Recurse to children bodies
        for child in body_elem.findall('body'):
            parse_body(child, body_name)

    # Start parsing from the first body inside worldbody
    for child in worldbody.findall('body'):
        parse_body(child)

    return bodies, joints


def print_model_structure(xml_path):
    if not os.path.exists(xml_path):
        print(f"Error: File not found at {xml_path}")
        return False

    bodies, joints = parse_model_structure(xml_path)

    print(f"\n{'='*60}")
    print(f" MODEL STRUCTURE: {os.path.basename(xml_path)}")
    print(f"{'='*60}")

    print(f"\n[CONTACT BODIES ORDER] (Total: {len(bodies)})")
    print(f"These correspond to the indices in the 'contacts' array.")
    for i, name in enumerate(bodies):
        print(f"  {i:2d}: {name}")

    print(f"\n[JOINT DOFS ORDER] (Total Joints: {len(joints)})")
    print(f"These correspond to 'frames[6:]' (after Root Pos/Rot).")

    dof_idx = 0
    for i, j in enumerate(joints):
        dof_range = f"{dof_idx}:{dof_idx+j['dim']}"
        print(f"  {i:2d}: {j['name']:<20} (Type: {j['type']:<10}, Dims: {j['dim']}, Local Indices: {dof_range})")
        dof_idx += j['dim']

    print(f"\nTotal Joint DOFs: {dof_idx}")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a MuJoCo character model's body and joint DOF order.")
    parser.add_argument("xml_path", nargs="?", default=None,
                        help="character .xml (default: {})".format(DEFAULT_XML))
    args = parser.parse_args()

    xml_path = args.xml_path
    if xml_path is None:
        xml_path = str(DEFAULT_XML)
        print(f"No XML file provided, using default: {xml_path}")

    raise SystemExit(0 if print_model_structure(xml_path) else 1)
