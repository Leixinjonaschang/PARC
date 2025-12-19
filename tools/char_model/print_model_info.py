import xml.etree.ElementTree as ET
import os
import sys

def print_model_structure(xml_path):
    if not os.path.exists(xml_path):
        print(f"Error: File not found at {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    bodies = []
    joints = []
    
    # Recursive function to parse bodies and joints
    def parse_body(body_elem, parent_name="world"):
        body_name = body_elem.attrib.get('name', 'unnamed')
        bodies.append(body_name)
        
        # Find joints in this body
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

if __name__ == "__main__":
    # Default path to humanoid.xml in PARC
    # Need to find where the script is running from to locate default assets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in PARC/tools/pkl_tools/
    # XML is in PARC/data/assets/humanoid.xml (up 3 levels, then down to data/assets)
    default_xml = os.path.join(script_dir, "../../../PARC/data/assets/humanoid.xml")
    
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = default_xml
        print(f"No XML file provided, using default: {xml_path}")
    
    print_model_structure(xml_path)

