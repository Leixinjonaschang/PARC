#!/usr/bin/env python3
"""
将 terrain-generator 生成的 OBJ 地形文件转换为 PARC 可用的 heightfield 格式。

使用方法:
    python obj_to_heightfield.py <input_obj_file> <output_pkl_file> [options]

示例:
    python obj_to_heightfield.py terrain.obj terrain.pkl --resolution 0.1 --device cpu
"""

import sys
import os
import argparse
import pickle
import numpy as np
import trimesh
import torch

# 设置路径以导入 PARC 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
parc_dir = os.path.abspath(os.path.join(script_dir, "../.."))
if parc_dir not in sys.path:
    sys.path.insert(0, parc_dir)

from util.terrain_util import SubTerrain

def get_heights_from_mesh(mesh: trimesh.Trimesh, origins: np.ndarray):
    """
    从 mesh 获取指定点的高度值（使用射线相交）。
    
    Args:
        mesh: trimesh.Trimesh 对象
        origins: (N, 3) 形状的数组，包含射线的起点 [x, y, z]
    
    Returns:
        heights: (N,) 形状的数组，包含每个点的高度值
    """
    # 如果 origins 只有 2 维，添加第 3 维（z 方向，从高处开始）
    if origins.shape[1] == 2:
        bbox = mesh.bounding_box.bounds
        max_z = np.max(bbox[:, 2])
        origins = np.concatenate([origins, np.ones((origins.shape[0], 1)) * (max_z + 10.0)], axis=1)
    
    # 创建向下的射线向量
    vectors = np.stack(
        [np.zeros_like(origins[:, 0]), np.zeros_like(origins[:, 1]), -np.ones_like(origins[:, 2])], 
        axis=-1
    )
    
    # 执行射线-网格相交查询
    points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    
    # 初始化高度数组
    heights = np.zeros(origins.shape[0])
    if len(points) > 0:
        heights[index_ray] = points[:, 2]
    
    return heights


def get_height_array_from_mesh(mesh: trimesh.Trimesh, resolution=0.1, border_offset=0.0):
    """
    从 mesh 生成高度数组。
    
    Args:
        mesh: trimesh.Trimesh 对象
        resolution: 网格分辨率（米）
        border_offset: 边界偏移量（米）
    
    Returns:
        height_array: (H, W) 形状的高度数组
        min_point: (2,) 形状的数组，包含 [min_x, min_y]
        center: (2,) 形状的数组，包含 [center_x, center_y]
    """
    # 获取 mesh 的边界框
    bbox = mesh.bounding_box.bounds
    b_min = np.min(bbox, axis=0)
    b_max = np.max(bbox, axis=0)
    center = (b_min + b_max) / 2
    
    # 计算网格尺寸
    dim_x = b_max[0] - b_min[0]
    dim_y = b_max[1] - b_min[1]
    
    # 计算网格点数
    n_points_x = int(dim_x / resolution) + 1
    n_points_y = int(dim_y / resolution) + 1
    
    # 生成网格点
    x = np.linspace(b_min[0] + border_offset, b_max[0] - border_offset, n_points_x)
    y = np.linspace(b_min[1] + border_offset, b_max[1] - border_offset, n_points_y)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    
    # 展平并创建射线起点
    origins = np.stack([
        xv.flatten(),
        yv.flatten(),
        np.ones_like(xv.flatten()) * (b_max[2] + 10.0)  # 从高处开始
    ], axis=-1)
    
    # 获取高度值
    heights = get_heights_from_mesh(mesh, origins)
    
    # 重塑为 2D 数组
    height_array = heights.reshape(n_points_x, n_points_y)
    
    min_point = np.array([b_min[0], b_min[1]])
    
    return height_array, min_point, center[:2]


def obj_to_heightfield(
    obj_path: str,
    output_path: str,
    resolution: float = 0.1,
    border_offset: float = 0.0,
    device: str = "cpu",
    terrain_name: str = "terrain"
):
    """
    将 OBJ 文件转换为 PARC heightfield 格式。
    
    Args:
        obj_path: 输入的 OBJ 文件路径
        output_path: 输出的 PKL 文件路径
        resolution: 网格分辨率（米），默认 0.1
        border_offset: 边界偏移量（米），默认 0.0
        device: 设备（"cpu" 或 "cuda:0"），默认 "cpu"
        terrain_name: 地形名称，默认 "terrain"
    """
    print(f"加载 OBJ 文件: {obj_path}")
    
    # 加载 mesh
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"文件不存在: {obj_path}")
    
    mesh = trimesh.load(obj_path)
    
    # 如果是 Scene 对象，提取第一个 mesh
    if isinstance(mesh, trimesh.Scene):
        print("检测到 Scene 对象，提取第一个 mesh...")
        mesh = list(mesh.geometry.values())[0]
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"无法从文件加载有效的 mesh: {obj_path}")
    
    print(f"Mesh 信息:")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  面数: {len(mesh.faces)}")
    print(f"  边界框: {mesh.bounding_box.bounds}")
    
    # 生成高度数组
    print(f"\n生成高度数组 (分辨率: {resolution}m)...")
    
    height_array, min_point, center = get_height_array_from_mesh(
        mesh,
        resolution=resolution,
        border_offset=border_offset
    )
    
    print(f"高度数组形状: {height_array.shape}")
    print(f"高度范围: [{np.min(height_array):.3f}, {np.max(height_array):.3f}]")
    print(f"最小点: {min_point}")
    print(f"中心点: {center}")
    
    # 创建 SubTerrain 对象
    print(f"\n创建 SubTerrain 对象...")
    x_dim, y_dim = height_array.shape
    dx = dy = resolution
    
    terrain = SubTerrain(
        terrain_name=terrain_name,
        x_dim=x_dim,
        y_dim=y_dim,
        dx=dx,
        dy=dy,
        min_x=float(min_point[0]),
        min_y=float(min_point[1]),
        device=device
    )
    
    # 设置高度场数据
    terrain.hf = torch.tensor(height_array, dtype=torch.float32, device=device)
    
    # 设置掩码（所有点都有效）
    terrain.hf_mask = torch.ones((x_dim, y_dim), dtype=torch.bool, device=device)
    
    # 设置最大最小高度（用于随机化）
    terrain.hf_maxmin[..., 0] = torch.tensor(np.max(height_array), dtype=torch.float32, device=device)
    terrain.hf_maxmin[..., 1] = torch.tensor(np.min(height_array), dtype=torch.float32, device=device)
    
    # 转换为 numpy 格式以便保存
    terrain.to_numpy()
    
    # 保存为 pickle 文件
    print(f"\n保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # PARC 期望的格式是 {"terrain": terrain}
    with open(output_path, 'wb') as f:
        pickle.dump({"terrain": terrain}, f)
    
    print("转换完成！")
    print(f"\n输出文件信息:")
    print(f"  文件路径: {output_path}")
    print(f"  地形尺寸: {x_dim} x {y_dim}")
    print(f"  分辨率: {dx}m x {dy}m")
    print(f"  实际尺寸: {x_dim * dx:.2f}m x {y_dim * dy:.2f}m")
    print(f"  最小点: [{terrain.min_point[0]:.3f}, {terrain.min_point[1]:.3f}]")
    
    return terrain


def main():
    parser = argparse.ArgumentParser(
        description="将 terrain-generator 生成的 OBJ 地形文件转换为 PARC 可用的 heightfield 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python obj_to_heightfield.py terrain.obj terrain.pkl
  
  # 指定分辨率
  python obj_to_heightfield.py terrain.obj terrain.pkl --resolution 0.05
  
  # 使用 GPU
  python obj_to_heightfield.py terrain.obj terrain.pkl --device cuda:0
        """
    )
    
    parser.add_argument("input_obj", type=str, help="输入的 OBJ 文件路径")
    parser.add_argument("output_pkl", type=str, help="输出的 PKL 文件路径")
    parser.add_argument(
        "--resolution", 
        type=float, 
        default=0.1,
        help="网格分辨率（米），默认 0.1"
    )
    parser.add_argument(
        "--border-offset",
        type=float,
        default=0.0,
        help="边界偏移量（米），默认 0.0"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备（'cpu' 或 'cuda:0'），默认 'cpu'"
    )
    parser.add_argument(
        "--terrain-name",
        type=str,
        default="terrain",
        help="地形名称，默认 'terrain'"
    )
    
    args = parser.parse_args()
    
    try:
        terrain = obj_to_heightfield(
            obj_path=args.input_obj,
            output_path=args.output_pkl,
            resolution=args.resolution,
            border_offset=args.border_offset,
            device=args.device,
            terrain_name=args.terrain_name
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

