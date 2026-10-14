"""Visualize a PARC terrain heightfield in 2D (matplotlib) or 3D (polyscope).

Accepts MS motion files (terrain_data), legacy terrain dicts ({"terrain": SubTerrain, ...},
as written by obj_to_heightfield.py, parc_2_kin_gen.py and develop-era motion files),
dicts with an "hf" array, and pickled SubTerrain objects.  Files are read with
parc.util.pickle_compat.

Usage, from anywhere:

    python scripts/tools/view_terrain.py <terrain .pkl> [--vis_type 2d|3d]
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np

# Setup the path to the repository root (scripts/tools -> repo)
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import parc.util.file_io as file_io
import parc.util.pickle_compat as pickle_compat
import parc.util.terrain_util as terrain_util


def _to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_ps_hf_mesh(hf, min_x, min_y, dx, name):
    """
    Creates a surface mesh in Polyscope from a heightfield.
    """
    import polyscope as ps

    # Convert PyTorch tensor to numpy if necessary
    hf = _to_numpy(hf)

    verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
        hf,
        min_x,
        min_y,
        dx
    )

    ps_mesh = ps.register_surface_mesh(name, verts, tris)
    ps_mesh.set_color([0.5, 0.5, 0.5]) # Grey color
    # Optional: Add height-based coloring
    ps_mesh.add_scalar_quantity("height", verts[:, 2], enabled=True, cmap="viridis")
    return ps_mesh

def extract_terrain_data(pkl_path):
    """
    Load pickle file and extract standard terrain properties.
    Returns: hf (numpy array), min_x, min_y, dx
    """
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return None, None, None, None

    print(f"Loading {pkl_path}...")
    data = pickle_compat.load(pkl_path)

    # 0. MS file: the terrain is its terrain_data section
    if file_io.is_ms_container(data):
        terrain_data = file_io.load_ms_file(pkl_path).terrain_data
        if terrain_data is None:
            print(f"No terrain_data in {pkl_path}")
            return None, None, None, None
        min_point = _to_numpy(terrain_data.min_point).reshape(-1)
        return _to_numpy(terrain_data.hf), float(min_point[0]), float(min_point[1]), float(terrain_data.dx)

    # 1. Extract Terrain Object
    if isinstance(data, dict) and "terrain" in data:
        terrain = data["terrain"]
    elif isinstance(data, dict) and "hf" in data:
        terrain = data
    else:
        # Some files might be the Terrain object directly
        terrain = data

    if terrain is None:
        print(f"No terrain in {pkl_path}")
        return None, None, None, None

    # 2. Extract Properties
    # Handle different object types (dict vs SubTerrain object)
    if isinstance(terrain, dict):
        hf = terrain["hf"]
        # Use defaults or extract if available
        # Assuming standard grid if not specified
        if "dxdy" in terrain:
            dx = _to_numpy(terrain["dxdy"]).reshape(-1)[0]
        else:
            dx = terrain.get("dx", 0.1)
        min_point = _to_numpy(terrain.get("min_point", [0.0, 0.0])).reshape(-1)
    elif hasattr(terrain, "hf"):
        # Assuming it's a SubTerrain object or similar class (torch or numpy fields)
        hf = terrain.hf
        dx = _to_numpy(terrain.dxdy).reshape(-1)[0]
        min_point = _to_numpy(terrain.min_point).reshape(-1)
    else:
        print(f"No terrain found in {pkl_path} (a {type(terrain).__name__})")
        return None, None, None, None

    # Final check to ensure hf is numpy
    return _to_numpy(hf), float(min_point[0]), float(min_point[1]), float(dx)

def view_terrain_2d(hf, title="Terrain Preview"):
    """Visualizes terrain heightfield in 2D using matplotlib"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(hf.T, cmap='terrain', origin='lower')
    plt.colorbar(label='Height (m)')
    plt.title(title)
    plt.xlabel("X Index")
    plt.ylabel("Y Index")
    print("Showing 2D view...")
    plt.show()

def view_terrain_3d(hf, min_x, min_y, dx):
    """Visualizes terrain heightfield in 3D using Polyscope"""
    import polyscope as ps

    # Initialize Polyscope
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0.1, 0.1, 0.1])

    # Register Mesh
    print(f"Building mesh for heightfield of shape {hf.shape}...")
    build_ps_hf_mesh(hf, min_x, min_y, dx, "terrain_mesh")

    # Show
    print("Showing 3D view. Close window to exit.")
    ps.show()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize PARC terrain files (MS files and legacy terrain pickles).")
    parser.add_argument("terrain_file", type=str, help="Path to the terrain .pkl file")
    parser.add_argument("--vis_type", type=str, choices=["2d", "3d"], default="3d",
                        help="Type of visualization: '2d' (matplotlib) or '3d' (polyscope). Default is 3d.")

    args = parser.parse_args(argv)

    hf, min_x, min_y, dx = extract_terrain_data(args.terrain_file)
    if hf is None:
        return 1

    print(f"Terrain: hf {hf.shape}, dx {dx:.4g}, min point ({min_x:.3f}, {min_y:.3f}), "
          f"heights [{np.min(hf):.3f}, {np.max(hf):.3f}]")
    if args.vis_type == "2d":
        view_terrain_2d(hf, title=f"Terrain Preview: {os.path.basename(args.terrain_file)}")
    else:
        view_terrain_3d(hf, min_x, min_y, dx)
    return 0

if __name__ == "__main__":
    sys.exit(main())
