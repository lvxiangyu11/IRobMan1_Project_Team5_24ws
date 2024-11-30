import os
import cv2
import numpy as np
import open3d as o3d
import csv
import copy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# 文件路径和相机内参
BASE_DIR = "./tmp"
CSV_FILE = os.path.join(BASE_DIR, "image_tf_record.csv")
CAMERA_INTRINSICS = {
    "fx": 700.0,
    "fy": 700.0,
    "cx": 640.0,
    "cy": 360.0,
    "scaling_factor": 1000.0
}

def load_tf_data(csv_file):
    """从 CSV 文件加载所有的 TF 数据"""
    tf_data_list = []
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                filename = row[0]
                translation = np.array([float(row[1]), float(row[2]), float(row[3])])
                rotation = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
                tf_data_list.append((filename, translation, rotation))
    except Exception as e:
        print(f"Failed to load TF data: {e}")
    return tf_data_list

def depth_to_pointcloud(depth_image, color_image, camera_intrinsics):
    """将深度图像和彩色图像转换为 3D 点云"""
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]
    scaling_factor = camera_intrinsics["scaling_factor"]

    height, width = depth_image.shape
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / scaling_factor  # 深度值 (单位: 米)
            if z == 0:  # 跳过无效深度点
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

            color = color_image[v, u][::-1] / 255.0  # BGR -> RGB, 并归一化到 [0, 1]
            colors.append(color)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
    pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pointcloud


def apply_tf_to_pointcloud(pointcloud, translation, rotation, mode='standard'):
    """
    将给定的平移和旋转应用到点云
    
    变换模式:
    - 'standard': 标准变换
    - 'inverse': 逆变换
    - 'camera_to_world': 相机到世界坐标系
    - 'world_to_camera': 世界到相机坐标系
    - 'symmetric': 对称变换
    """
    print(f"\n🔍 变换信息:")
    print(f"平移向量: {translation}")
    print(f"旋转四元数: {rotation}")

    # 创建变换矩阵
    transform = np.eye(4)
    
    # 创建旋转矩阵
    rot_matrix = R.from_quat(rotation).as_matrix()

    if mode == 'standard':
        # 标准变换
        transform[:3, 3] = translation
        transform[:3, :3] = rot_matrix
        print("🔧 使用标准变换")
    
    elif mode == 'inverse':
        # 完全逆变换
        inv_rot_matrix = np.linalg.inv(rot_matrix)
        inv_translation = -inv_rot_matrix @ translation
        
        transform[:3, 3] = inv_translation
        transform[:3, :3] = inv_rot_matrix
        print("🔧 使用完全逆变换")
    
    elif mode == 'camera_to_world':
        # 相机坐标系转世界坐标系
        transform[:3, 3] = translation
        transform[:3, :3] = rot_matrix
        print("🔧 相机坐标系到世界坐标系")
    
    elif mode == 'world_to_camera':
        # 世界坐标系转相机坐标系
        inv_rot_matrix = rot_matrix.T
        inv_translation = -inv_rot_matrix @ translation
        
        transform[:3, 3] = inv_translation
        transform[:3, :3] = inv_rot_matrix
        print("🔧 世界坐标系到相机坐标系")
    
    elif mode == 'symmetric':
        # 对称变换 - 考虑坐标系对称性
        sym_rot_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]) @ rot_matrix
        sym_translation = np.array([
            -translation[0],
            -translation[1],
            translation[2]
        ])
        
        transform[:3, 3] = sym_translation
        transform[:3, :3] = sym_rot_matrix
        print("🔧 对称变换")
    
    else:
        raise ValueError("无效的变换模式")
    
    # 打印变换矩阵
    print("\n📊 变换矩阵:")
    print(transform)

    # 应用变换
    pointcloud.transform(transform)
    return pointcloud


def main():
    # 从 CSV 文件中加载 TF 数据
    tf_data_list = load_tf_data(CSV_FILE)

    # 尝试不同的变换模式
    modes = [
        'standard', 
        'inverse', 
        'camera_to_world', 
        'world_to_camera', 
        'symmetric'
    ]

    # 对每种模式进行处理
    for mode in modes:
        # 创建一个空的点云用于存储所有的点
        all_pointcloud = o3d.geometry.PointCloud()

        # 使用 tqdm 显示进度条
        for filename, translation, rotation in tqdm(tf_data_list, desc=f"Processing images - {mode}", unit="image"):
            # 只处理深度图文件
            if "depth" not in filename:
                continue

            depth_filepath = os.path.join(BASE_DIR, "depths", filename)
            color_filepath = os.path.join(BASE_DIR, "colors", filename.replace("depth", "color"))

            # 加载深度图和彩色图
            depth_image = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Error: Unable to load depth image from {depth_filepath}.")
                continue

            color_image = cv2.imread(color_filepath, cv2.IMREAD_COLOR)
            if color_image is None:
                print(f"Error: Unable to load color image from {color_filepath}.")
                continue

            # 将深度图和彩色图生成点云
            pointcloud = depth_to_pointcloud(depth_image, color_image, CAMERA_INTRINSICS)

            try:
                # 应用特定模式的变换
                transformed_pointcloud = apply_tf_to_pointcloud(
                    pointcloud, 
                    translation, 
                    rotation, 
                    mode=mode
                )
                
                # 将当前点云合并到总点云中
                all_pointcloud += transformed_pointcloud
                
            except Exception as e:
                print(f"模式 {mode} 出错: {e}")

        # 可视化当前模式下的所有点云
        print(f"\n正在显示 {mode} 模式的点云")
        o3d.visualization.draw_geometries([all_pointcloud], window_name=f"点云变换 - {mode}")

if __name__ == "__main__":
    main()