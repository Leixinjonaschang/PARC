run scripts/motionscope.py in the root directory.

edit the config in parc/motionscope/motionscope_config.yaml to load different motions, mdm models, char models, etc.

Each `motions` entry is `[motion file, character]`, where the character is `humanoid`, `g1` (Unitree G1, `data/assets/unitree_g1/g1_mocap_29dof.xml`) or a path to a character xml. Motion files must be MS files; convert legacy ones with scripts/convert_legacy_motions.py.
