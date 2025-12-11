import sys
import os
import pickle
import numpy as np
import torch
import trimesh
import polyscope as ps
import matplotlib.pyplot as plt
import argparse

# Setup the path to the PARC directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# PARC/tools/terrain_tool -> PARC/
parc_dir = os.path.abspath(os.path.join(script_dir, "../.."))
if parc_dir not in sys.path:
    sys.path.insert(0, parc_dir)

import util.terrain_util as terrain_util

def build_ps_hf_mesh(hf, min_x, min_y, dx, name):
    """
    Creates a surface mesh in Polyscope from a heightfield.
    """
    # Convert PyTorch tensor to numpy if necessary
    if hasattr(hf, 'detach'):
        hf = hf.detach().cpu().numpy()
    
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
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 1. Extract Terrain Object
    if "terrain" in data:
        terrain = data["terrain"]
    elif isinstance(data, dict) and "hf" in data: 
        terrain = data
    else:
        # Some files might be the Terrain object directly
        terrain = data

    # 2. Extract Properties
    # Handle different object types (dict vs SubTerrain object)
    if isinstance(terrain, dict):
        hf = terrain["hf"]
        # Use defaults or extract if available
        # Assuming standard grid if not specified
        dx = terrain.get("dxdy", [0.1, 0.1])[0] 
        min_x = terrain.get("min_point", [0.0, 0.0])[0]
        min_y = terrain.get("min_point", [0.0, 0.0])[1]
    else:
        # Assuming it's a SubTerrain object or similar class
        # Ensure it's on CPU
        if hasattr(terrain, "to_numpy"):
             try:
                terrain.to_numpy()
             except AttributeError:
                pass # Already numpy
        
        hf = terrain.hf
        # Handle tensor vs numpy for properties
        if hasattr(terrain.dxdy, "item"):
             dx = terrain.dxdy[0].item()
        else:
             dx = terrain.dxdy[0]

        if hasattr(terrain.min_point, "cpu"):
             min_x = terrain.min_point[0].item()
             min_y = terrain.min_point[1].item()
        else:
             min_x = terrain.min_point[0]
             min_y = terrain.min_point[1]
             
    # Final check to ensure hf is numpy
    if hasattr(hf, 'detach'):
        hf = hf.detach().cpu().numpy()
        
    return hf, min_x, min_y, dx

def view_terrain_2d(hf, title="Terrain Preview"):
    """Visualizes terrain heightfield in 2D using matplotlib"""
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PARC terrain files.")
    parser.add_argument("terrain_file", type=str, help="Path to the terrain .pkl file")
    parser.add_argument("--vis_type", type=str, choices=["2d", "3d"], default="3d", 
                        help="Type of visualization: '2d' (matplotlib) or '3d' (polyscope). Default is 3d.")
    
    args = parser.parse_args()
    
    hf, min_x, min_y, dx = extract_terrain_data(args.terrain_file)
    
    if hf is not None:
        if args.vis_type == "2d":
            view_terrain_2d(hf, title=f"Terrain Preview: {os.path.basename(args.terrain_file)}")
        else:
            view_terrain_3d(hf, min_x, min_y, dx)
