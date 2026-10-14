import time

import isaacgym.gymapi as gymapi
import numpy as np
import torch


def add_trimesh_to_gym(verts: np.ndarray, tris: np.ndarray, 
                       sim_obj, gym_obj,
                       x_offset = 0.0, y_offset = 0.0,
                       static_friction = 1.0, dynamic_friction = 1.0, restitution = 0.0):
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = verts.shape[0]
    tm_params.nb_triangles = tris.shape[0]
    tm_params.transform.p.x = x_offset
    tm_params.transform.p.y = y_offset
    tm_params.static_friction = static_friction
    tm_params.dynamic_friction = dynamic_friction
    tm_params.restitution = restitution
    print("adding triangle mesh to gym")
    start_time = time.perf_counter()
    gym_obj.add_triangle_mesh(sim_obj, verts.flatten(), tris.flatten(), tm_params)
    end_time = time.perf_counter()
    print("adding mesh to gym time:", end_time-start_time, " seconds.")
    return