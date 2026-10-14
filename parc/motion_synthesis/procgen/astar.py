import heapq
import random
import time
from typing import Optional

import numpy as np
import torch

import parc.util.terrain_util as terrain_util


class AStarSettings:
    max_z_diff = 2.1
    max_jump_xy_dist = 3.0
    max_jump_z_diff = 0.3
    min_jump_z_diff = -0.7
    w_z = 0.15
    w_xy = 1.0
    w_bumpy = 1.0
    max_bumpy = 0.2
    uniform_cost_max = 0.25
    uniform_cost_min = 0.0
    min_start_end_xy_dist = 4.0
    max_cost = 1000.0

class TerrainNode(object):
    def __init__(self, index: np.ndarray, pos: np.ndarray, is_border: bool):
        self.index = index
        self.pos = pos
        self.edges = []
        self.is_border = is_border
        return

    def __repr__(self):
            return f'{self.index}'

    def __lt__(self, other):
        if isinstance(other, TerrainNode):
            return (self.index < other.index).any()
        return False
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, TerrainNode):
            return (self.index == other.index).all()
        return False

    def key(self):
        return self.index.tobytes()
    
def pick_random_start_end_nodes(terrain: terrain_util.SubTerrain,
                                min_dist = 4.0, max_attempts=1000):
    device = terrain.hf.device
    for _ in range(max_attempts):
        start_grid_ind = [
            torch.randint(low=0, high=terrain.dims[0].item(), size=[1], device=device),
            torch.randint(low=0, high=terrain.dims[1].item(), size=[1], device=device)
        ]
        start_grid_ind = torch.cat(start_grid_ind)
        end_grid_ind = [
            torch.randint(low=0, high=terrain.dims[0].item(), size=[1], device=device),
            torch.randint(low=0, high=terrain.dims[1].item(), size=[1], device=device)
        ]
        end_grid_ind = torch.cat(end_grid_ind)

        path_dist = torch.norm(terrain.get_point(start_grid_ind) - terrain.get_point(end_grid_ind))
        if path_dist >= min_dist - 1e-4:
            print("path dist:", path_dist.item())
            return start_grid_ind, end_grid_ind
    
    assert False, "max attempts to find far enough start/end nodes"
    return

def pick_random_start_end_nodes_on_edges(terrain: terrain_util.SubTerrain,
                                         min_dist = 7.0, max_attempts=1000):
    device = terrain.hf.device
    near_border_indices = []
    for i in range(terrain.hf.shape[0]):
        for j in range(terrain.hf.shape[1]):
            if (i == 1 or i == 2 or i == terrain.hf.shape[0] - 2 or i == terrain.hf.shape[0] - 3) or \
                (j == 1 or j == 2 or j == terrain.hf.shape[1] - 2 or j == terrain.hf.shape[1] - 3):
                near_border_indices.append(torch.tensor([i, j], dtype=torch.int64))

    near_border_indices = torch.stack(near_border_indices)
    for _ in range(max_attempts):
        start_grid_ind = near_border_indices[random.randint(0, near_border_indices.shape[0]-1)]
        end_grid_ind = near_border_indices[random.randint(0, near_border_indices.shape[0]-1)]

        start_grid_ind = start_grid_ind.to(device=device)
        end_grid_ind = end_grid_ind.to(device=device)
        path_dist = torch.norm(terrain.get_point(start_grid_ind) - terrain.get_point(end_grid_ind))
        if path_dist >= min_dist - 1e-4:
            print("path dist:", path_dist.item())
            return start_grid_ind, end_grid_ind
        
    assert False, "max attempts to find far enough start/end nodes"
    return

def construct_navigation_graph(terrain: terrain_util.SubTerrain,
                               max_z_diff: float = 2.1,
                               max_jump_xy_dist: float = 3.0,
                               max_jump_z_diff: float = 0.3,
                               min_jump_z_diff: float = -1.0):
    # each cell is a node
    # connect edges between adjacent cells, with 1 param describing xy distance, and another describing z dist
    # all edges will be directed for simplicity?

    # After all these initial edges are made, we need to add special edges to allow for jumping over gaps
    # Every cell has the potential to jump to far away cells
    # Connect each cell with the cells within radius R
    # remove edges that already exist in the graph (e.g.)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    cross_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    device = terrain.hf.device
    grid_dim_x = terrain.dims[0].item()
    grid_dim_y = terrain.dims[1].item()

    def in_bounds(row, col):
        return row >= 0 and row <= grid_dim_x - 1 and col >= 0 and col <= grid_dim_y - 1

    nodes = []
    for i in range(grid_dim_x):
        nodes.append([])
        for j in range(grid_dim_y):
            grid_inds = torch.tensor([i, j], dtype=torch.int64, device=device)
            pos = terrain.get_xyz_point(grid_inds)

            is_border = i == 0 or i == grid_dim_x-1 or j==0 or j==grid_dim_y-1

            new_node = TerrainNode(grid_inds.cpu().numpy(), pos.cpu().numpy(), is_border)
            nodes[i].append(new_node)

    # Now add edges between adjacent nodes provided that they are within the acceptable height
    for i in range(grid_dim_x):
        for j in range(grid_dim_y):
            curr_node = nodes[i][j]
            for drow, dcol in directions:
                row = i + drow
                col = j + dcol
                if in_bounds(row, col):
                    other_node = nodes[row][col]

                    # NOTE: missing edge case: if its a corner node, also check that the two adjacent nodes don't block it
                    
                    if abs(other_node.pos[2] - curr_node.pos[2]) <= max_z_diff:
                        curr_node.edges.append((row, col))

    # A node is a cliff node if:
    #   1. the 4 cross nodes are same or lower height than center node
    #   2. Not a border node

    def is_cliff_node(i, j):
        curr_node = nodes[i][j]
        if curr_node.is_border:
            return False
        
        for drow, dcol in cross_directions:
            other_node = nodes[i+drow][j+dcol]
            if curr_node.pos[2] - other_node.pos[2] > 1e-3:
                return True
        return False

    #cliff_nodes = []
    dx = terrain.dxdy[0].item()
    node_search_radius = int(np.ceil(max_jump_xy_dist / dx))
    for i in range(grid_dim_x):
        for j in range(grid_dim_y):
            curr_node = nodes[i][j]
            if is_cliff_node(i, j):
                # Search for other cliff nodes in a bounding box around this node

                row_search_start = max(i-node_search_radius, 1)
                row_search_end = min(i+node_search_radius, terrain.hf.shape[0]-1)

                col_search_start = max(j-node_search_radius, 1)
                col_search_end = min(j+node_search_radius, terrain.hf.shape[1]-1)

                for ii in range(row_search_start, row_search_end):
                    for jj in range(col_search_start, col_search_end):
                        if is_cliff_node(ii, jj):

                            other_node = nodes[ii][jj]
                            if np.linalg.norm(curr_node.pos[0:2] - other_node.pos[0:2]) <= max_jump_xy_dist and \
                                min_jump_z_diff <= other_node.pos[2] - curr_node.pos[2] <= max_jump_z_diff:


                                # Check if there are walls in the way
                                # First construct a ray from curr_node to other_node
                                # Then check the heights of the intermediate cells
                                # If one cell has a height > curr_node height + jump height, then there is a wall blocking the way
                                line_indices = terrain_util.get_line_indices(x0=curr_node.index[0].item(),
                                                                            y0=curr_node.index[1].item(),
                                                                            x1 = other_node.index[0].item(),
                                                                            y1 = other_node.index[1].item())
                                line_indices = torch.tensor(line_indices, dtype=torch.int64, device=device)
                                line_indices = terrain.get_inbounds_grid_index(line_indices)
                                line_heights = terrain.hf[line_indices[:, 0], line_indices[:, 1]]

                                if (line_heights < curr_node.pos[2] + max_jump_z_diff + 1e-3).all():

                                    curr_node.edges.append((ii, jj))
    return nodes

def a_star_search(terrain: terrain_util.SubTerrain,
                   start: np.ndarray, goal: np.ndarray,
                   max_z_diff: float = 2.1,
                   max_jump_xy_dist: float = 3.0,
                   max_jump_z_diff: float = 0.3,
                   min_jump_z_diff: float = -1.0,
                   w_z: float = 1.0,
                   w_xy: float = 1.0,
                   w_bumpy: float = 1.0,
                   max_bumpy: float = 0.2,
                   stochastic_step_cost_fn = None,
                   nav_graph: Optional[list] = None,
                   max_compute_time = 1.0, # seconds
                   verbose = False):
    

    if verbose:
        construct_graph_start_time = time.time()

    if nav_graph is None:
        nav_graph = construct_navigation_graph(terrain=terrain, max_z_diff=max_z_diff, 
                                        max_jump_xy_dist=max_jump_xy_dist, 
                                        max_jump_z_diff=max_jump_z_diff,
                                        min_jump_z_diff=min_jump_z_diff)
    compute_start_time = time.time()

    if verbose:
        construct_graph_end_time = time.time()
        print("construct graph time:", construct_graph_end_time - construct_graph_start_time)

    def compute_bumpy_cost(node: TerrainNode):

        #center_h = terrain.hf[node.index[0], node.index[1]]

        # Mean Absolute Difference between shifted 3x3 patches

        # h00 h01 h02
        # h10 h11 h12
        # h20 h21 h22
        grid_inds_x = torch.tensor([-1, 0, 1], dtype=torch.int64, device=terrain.hf.device)
        grid_inds_y = grid_inds_x.clone()
        grid_inds_x, grid_inds_y = torch.meshgrid(grid_inds_x, grid_inds_y)

        grid_inds_x = grid_inds_x + node.index[0].item()
        grid_inds_y = grid_inds_y + node.index[1].item()

        clamped_grid_inds_x = torch.clamp(grid_inds_x, min=0, max=terrain.hf.shape[0]-1)
        clamped_grid_inds_y = torch.clamp(grid_inds_y, min=0, max=terrain.hf.shape[1]-1)

        center_h = terrain.hf[clamped_grid_inds_x, clamped_grid_inds_y]

        mad = 0.0
        for i in range(-1, 2):
            for j in range(-1, 2):
                clamped_grid_inds_x = torch.clamp(grid_inds_x + i, min=0, max=terrain.hf.shape[0]-1)
                clamped_grid_inds_y = torch.clamp(grid_inds_y + j, min=0, max=terrain.hf.shape[1]-1)

                h = terrain.hf[clamped_grid_inds_x, clamped_grid_inds_y]

                mad += torch.sum(torch.abs(center_h - h)).item()

        mad = mad / 81

        return mad

    def cost(from_node: TerrainNode, to_node: TerrainNode):
        """Calculate the cost to move from one position to another based on height difference."""

        abs_height_diff = abs(to_node.pos[2] - from_node.pos[2])
        z_cost = w_z * abs_height_diff * abs_height_diff # square prioritizes more shorter height diffs along path


        xy_diff = to_node.pos[0:2] - from_node.pos[0:2]
        xy_dist = np.sum(xy_diff * xy_diff) # square prioritizes walking around rather than jumping across gaps
        xy_cost = w_xy * xy_dist


        bumpy_cost = compute_bumpy_cost(to_node)
        
        if bumpy_cost > max_bumpy:
            bumpy_cost = max_bumpy

        bumpy_cost *= w_bumpy

        total_cost = xy_cost + z_cost + bumpy_cost

        if stochastic_step_cost_fn is not None:
            total_cost += stochastic_step_cost_fn()
        return total_cost

    def heuristic(a: TerrainNode, b: TerrainNode):
        """Heuristic function for A*, using Euclidean distance."""
        return np.linalg.norm(a.pos - b.pos)

    if verbose:
        a_star_start_time = time.time()

    # Priority queue for the A* algorithm
    open_set = []
    start_node = nav_graph[start[0]][start[1]]
    goal_node = nav_graph[goal[0]][goal[1]]
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, start_node))
    
    # Maps to store cost and parent info
    g_score = {start_node.key(): 0}
    came_from = {}
    
    while open_set:
        _, current_cost, curr_node = heapq.heappop(open_set)
        
        # Check if we reached the goal
        curr_compute_time = time.time() - compute_start_time
        out_of_time = curr_compute_time > max_compute_time
        if curr_node == goal_node or out_of_time:

            if out_of_time:
                print("out of time")

            total_cost = g_score[curr_node.key()]
            

            path = []
            while curr_node.key() in came_from:
                path.append(curr_node.index)
                curr_node = came_from[curr_node.key()]
            path.append(start_node.index)

            if verbose:
                a_star_end_time = time.time()
                print("a_star time:", a_star_end_time - a_star_start_time)
                print("Best path cost:", total_cost)

            return path[::-1], total_cost  # Return reversed path
        
        # Explore neighbors
        for edge in curr_node.edges:
            next_node = nav_graph[edge[0]][edge[1]]
            tentative_g_score = g_score[curr_node.key()] + cost(curr_node, next_node)
            if next_node.key() not in g_score or tentative_g_score < g_score[next_node.key()]:
                g_score[next_node.key()] = tentative_g_score
                f_score = tentative_g_score + heuristic(next_node, goal_node)
                heapq.heappush(open_set, (f_score, tentative_g_score, next_node))
                came_from[next_node.key()] = curr_node

            # TODO: explore jumping mode?
            # if neighbor height < current height, then keep track of original cell and try to find a cell with same height
    
    if verbose:
        a_star_end_time = time.time()
        print("a_star time:", a_star_end_time - a_star_start_time)
        print("no path found")

    return None, None  # Return None if no path is found

def run_a_star_on_start_end_nodes(terrain: terrain_util.SubTerrain, 
                                  start_node, end_node, 
                                  settings: AStarSettings,
                                  nav_graph = None
    ):

    uniform_cost_min = settings.uniform_cost_min
    uniform_cost_max = settings.uniform_cost_max
    max_z_diff = settings.max_z_diff
    max_jump_xy_dist = settings.max_jump_xy_dist
    max_jump_z_diff = settings.max_jump_z_diff
    min_jump_z_diff = settings.min_jump_z_diff
    w_z = settings.w_z
    w_xy = settings.w_xy
    w_bumpy = settings.w_bumpy
    max_bumpy = settings.max_bumpy
    max_cost = settings.max_cost

    def uniform_cost():
        return np.random.random() * (uniform_cost_max - uniform_cost_min) + uniform_cost_min

    path_nodes, cost = a_star_search(terrain, start = start_node, 
                                     goal = end_node,
                                     max_z_diff = max_z_diff,
                                     max_jump_xy_dist = max_jump_xy_dist,
                                     max_jump_z_diff = max_jump_z_diff,
                                     min_jump_z_diff = min_jump_z_diff,
                                     w_z = w_z,
                                     w_xy = w_xy,
                                     w_bumpy = w_bumpy,
                                     max_bumpy = max_bumpy,
                                     stochastic_step_cost_fn=uniform_cost,
                                     nav_graph = nav_graph)
    if path_nodes is None:
        return False

    if cost > max_cost:
        return False
    
    th_path_nodes = []
    for node in path_nodes:
        th_path_nodes.append(torch.from_numpy(node).to(dtype=torch.int64, device=terrain.hf.device))

    
    nodes = torch.stack(th_path_nodes)


    # switch to saving path nodes as 3d posititons
    path_nodes_3dpos = []

    num_nodes = nodes.shape[0]
    curr_node_pos = terrain.get_xyz_point(nodes[0])
    path_nodes_3dpos.append(curr_node_pos.unsqueeze(0))

    split_dist = np.sqrt(terrain.dxdy[0].item() ** 2 + terrain.dxdy[1].item() ** 2) + 1e-3

    for i in range(1, num_nodes):
        prev_node_pos = curr_node_pos
        curr_node_pos = terrain.get_xyz_point(nodes[i])


        # if xy distance is large, split it up
        xy_dist = curr_node_pos[0:2] - prev_node_pos[0:2]
        xy_dist = torch.linalg.norm(xy_dist).item()

        if xy_dist > split_dist:

            steps = int(np.ceil(xy_dist / terrain.dxdy[0].item()))
            # split this path into nodes of approximately length dx
            x_pos = torch.linspace(prev_node_pos[0], curr_node_pos[0], steps=steps)
            y_pos = torch.linspace(prev_node_pos[1], curr_node_pos[1], steps=steps)
            z_pos = torch.linspace(prev_node_pos[2], curr_node_pos[2], steps=steps)
            xyz_pos = torch.cat([x_pos.unsqueeze(-1), y_pos.unsqueeze(-1), z_pos.unsqueeze(-1)], dim=1)
            xyz_pos = xyz_pos[1:].to(device=terrain.hf.device)
            path_nodes_3dpos.append(xyz_pos)
        else:
            path_nodes_3dpos.append(curr_node_pos.unsqueeze(0))


    path_nodes_3dpos = torch.cat(path_nodes_3dpos, dim=0)
    return path_nodes_3dpos


def catmull_rom_path(terrain: terrain_util.SubTerrain, control_points = None, num_samples=200):
    def catmull_rom_spline(control_points, num_samples):
        """
        Generate Catmull-Rom spline from control points.
        
        Args:
        - control_points (np.array): Array of control points of shape (M, 2), where M is the number of control points.
        - num_samples (int): Number of points to sample along the spline.
        
        Returns:
        - spline_points (np.array): Array of sampled points along the spline of shape (N, 2).
        """
        def catmull_rom(p0, p1, p2, p3, t):
            """
            Compute the Catmull-Rom spline for given control points and parameter t.
            """
            t2 = t * t
            t3 = t2 * t
            a1 = -0.5 * t + t2 - 0.5 * t3
            a2 = 1.0 - 2.5 * t2 + 1.5 * t3
            a3 = 0.5 * t + 2.0 * t2 - 1.5 * t3
            a4 = -0.5 * t2 + 0.5 * t3
            
            x = a1 * p0[0] + a2 * p1[0] + a3 * p2[0] + a4 * p3[0]
            y = a1 * p0[1] + a2 * p1[1] + a3 * p2[1] + a4 * p3[1]
            
            return np.array([x, y])
        
        # Prepare the output list
        spline_points = []
        
        # Number of control points
        M = len(control_points)
        
        # Iterate through the control points and generate spline segments
        for i in range(1, M - 2):
            p0 = control_points[i-1]
            p1 = control_points[i]
            p2 = control_points[i+1]
            p3 = control_points[i+2]
            
            # Sample points between p1 and p2
            for t in np.linspace(0, 1, num_samples):
                point = catmull_rom(p0, p1, p2, p3, t)
                spline_points.append(point)
        
        # Convert list to a numpy array and return
        spline_points = np.array(spline_points)

            # Calculate the length of the spline
        spline_length = 0
        for i in range(1, len(spline_points)):
            spline_length += np.linalg.norm(spline_points[i] - spline_points[i-1])
        
        return spline_points, spline_length

    # Pick 6 random points on the terrain
    if control_points is None:
        control_points = []
        for i in range(6):
            max_point = terrain.get_max_point()
            min_point = terrain.min_point
            x_pos = np.random.random() * (max_point[0].item() - min_point[0].item()) + min_point[0].item()
            y_pos = np.random.random() * (max_point[1].item() - min_point[1].item()) + min_point[1].item()
            control_points.append([x_pos, y_pos])

    #control_points = [[0.0, 0.0], [1.2, 1.2], [6, 3.6], [5.2, 6.8], [7.6, 11.6], [12.4, 12.4]]

    path_points, spline_length = catmull_rom_spline(control_points=control_points, num_samples=num_samples)
    num_samples = int(spline_length / terrain.dxdy[0].item())
    path_points, spline_length = catmull_rom_spline(control_points=control_points, num_samples=num_samples)
    #print(num_samples)
    path_points = torch.from_numpy(path_points).to(dtype=torch.float32)
    path_heights = terrain.get_hf_val_from_points(path_points)
    path_points = torch.cat([path_points, path_heights.unsqueeze(-1)], dim=-1)

    return path_points

def straight_line(terrain: terrain_util.SubTerrain, start_node = None, end_node = None):

    if start_node is None or end_node is None:
        min_dist_x = terrain.dxdy[0].item() * terrain.dims[0].item()
        min_dist_y = terrain.dxdy[1].item() * terrain.dims[1].item()
        min_dist = min(min_dist_x, min_dist_y) - terrain.dxdy[0].item() * 4
        start_node, end_node = pick_random_start_end_nodes_on_edges(terrain, min_dist = min_dist)

    start_node_2d_pos = terrain.get_point(start_node)
    end_node_2d_pos = terrain.get_point(end_node)

    # start_node_3d_pos = terrain.get_xyz_point(torch.from_numpy(g.PathPlanningSettings().start_node).to(dtype=torch.int64))
    # end_node_3d_pos = terrain.get_xyz_point(torch.from_numpy(g.PathPlanningSettings().end_node).to(dtype=torch.int64))

    num_points = torch.norm(start_node_2d_pos - end_node_2d_pos).item() / terrain.dxdy[0].item()
    num_points = int(num_points)

    # Create a linear space of interpolation factors
    t = torch.linspace(0, 1, steps=num_points).unsqueeze(1)

    path_points_xy = (1 - t) * start_node_2d_pos + t * end_node_2d_pos
    path_heights = terrain.get_hf_val_from_points(path_points_xy)
    path_points_xyz = torch.cat([path_points_xy, path_heights.unsqueeze(-1)], dim=-1)
    return path_points_xyz