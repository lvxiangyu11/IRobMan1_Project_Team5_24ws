import cv2
import numpy as np
import open3d as o3d

# ZED2 相机内参
CAMERA_INTRINSICS = {
    "fx": 700.0,  # 替换为你的相机参数
    "fy": 700.0,  # 替换为你的相机参数
    "cx": 640.0,  # 替换为你的相机参数
    "cy": 360.0,  # 替换为你的相机参数
    "scaling_factor": 1000.0  # 深度图的比例因子
}

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

def visualize_point_cloud(pointcloud):
    """可视化点云"""
    o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud Visualization")

def main():
    # 深度图和彩色图的路径
    depth_image_path = "/opt/ros_ws/tmp/depths/depth_4727967000000.png"
    color_image_path = "/opt/ros_ws/tmp/colors/color_4727934000000.png"

    # 读取深度图和彩色图
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"无法加载深度图像: {depth_image_path}")
        return

    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    if color_image is None:
        print(f"无法加载彩色图像: {color_image_path}")
        return

    # 生成点云
    pointcloud = depth_to_pointcloud(depth_image, color_image, CAMERA_INTRINSICS)
    
    # 保存点云到文件（可选）
    o3d.io.write_point_cloud("output_point_cloud.ply", pointcloud)
    print("点云已保存到 output_point_cloud.ply")

    # 可视化点云
    visualize_point_cloud(pointcloud)

if __name__ == "__main__":
    main()
