"""Character identifiers accepted in the `motions` entries of motionscope_config.yaml.

Kept out of include/global_header.py, which initialises polyscope on import, so that it
can be tested without a display.
"""

CHAR_PATHS = {
    "humanoid": "data/assets/humanoid.xml",
    "g1": "data/assets/unitree_g1/g1_mocap_29dof.xml",
}


def resolve_char_path(char_identifier: str) -> str:
    """Returns the character file of a known identifier (case-insensitive); any other
    value is taken to be a character file path itself."""
    return CHAR_PATHS.get(char_identifier.lower(), char_identifier)
