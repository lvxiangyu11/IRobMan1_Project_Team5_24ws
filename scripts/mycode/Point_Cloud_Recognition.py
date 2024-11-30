import os
import numpy as np
import open3d as o3d
import random

def load_point_cloud(file_path):
    """加载点云文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"点云文件未找到: {file_path}")
    pointcloud = o3d.io.read_point_cloud(file_path)
    print(f"成功加载点云: {file_path}, 包含 {len(pointcloud.points)} 个点")

    pointcloud = filter_point_cloud_depth(pointcloud)
    return pointcloud

def filter_point_cloud_depth(pointcloud, min_depth=-0.001):
    """
    过滤点云中深度小于 min_depth 的点
    :param pointcloud: 输入的 Open3D 点云对象
    :param min_depth: 最小深度阈值，单位为米
    :return: 过滤后的点云
    """
    # 获取点云的 numpy 坐标数组
    points = np.asarray(pointcloud.points)
    
    # 根据 Z 轴（深度）过滤点云
    filtered_points = points[points[:, 2] >= min_depth]
    
    # 创建新的点云对象
    filtered_pointcloud = o3d.geometry.PointCloud()
    filtered_pointcloud.points = o3d.utility.Vector3dVector(filtered_points)
    
    # 如果点云有颜色数据，同步保留对应颜色
    if pointcloud.has_colors():
        colors = np.asarray(pointcloud.colors)
        filtered_colors = colors[points[:, 2] >= min_depth]
        filtered_pointcloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"过滤完成，剩余点数: {len(filtered_points)}")
    return filtered_pointcloud

def load_obj_with_open3d(obj_path, translation, scale_factor=1):
    """
    使用 Open3D 加载 OBJ 模型并转换为点云
    :param obj_path: OBJ 文件路径
    :param translation: 平移量，用于布局模型
    :param scale_factor: 缩放因子（将 cm 转换为 m）
    :return: Open3D 点云
    """
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
        # mesh.scale(scale_factor, center=(0, 0, 0))  # 缩放单位从 cm 转换为 m
        mesh.translate(translation)  # 应用平移
        pointcloud = mesh.sample_points_uniformly(number_of_points=10000)  # 从网格采样点
        print(f"成功加载模型: {obj_path}, 转换为点云包含 {len(pointcloud.points)} 个点")
        return pointcloud
    except Exception as e:
        print(f"无法加载模型 {obj_path}: {e}")
        return None

def load_obj_models(obj_dir, spacing=1.0):
    """
    使用 Open3D 加载多个 OBJ 模型
    :param obj_dir: OBJ 文件目录
    :param spacing: 每个模型之间的间隔距离
    :return: 模型点云列表
    """
    obj_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith('.obj')]
    models = []
    for i, obj_file in enumerate(obj_files):
        translation = [i * spacing, 0, 0]  # 设置平移量
        model = load_obj_with_open3d(obj_file, translation)
        if model:
            models.append(model)
    return models

def get_random_initial_transformation(pointcloud, z_threshold=0.03):
    """
    从点云中随机选择一个 Z 轴大于指定阈值的点作为初始变换的平移量
    :param pointcloud: 输入的点云
    :param z_threshold: Z 轴的阈值
    :return: 初始变换矩阵 (4x4)
    """
    points = np.asarray(pointcloud.points)
    valid_points = points[points[:, 2] > z_threshold]

    if len(valid_points) == 0:
        raise ValueError("未找到满足 Z > {} 的点".format(z_threshold))

    # 随机选择一个点
    selected_point = valid_points[np.random.choice(len(valid_points))]
    print(f"随机选取的初始点: {selected_point}")

    # 构造初始变换矩阵
    initial_transform = np.eye(4)
    initial_transform[:3, 3] = selected_point
    return initial_transform

def match_multiple_instances(pointcloud, model, max_distance, min_fitness=0.3, max_iterations=1000):
    remaining_cloud = pointcloud
    results = []

    for iteration in range(max_iterations):
        try:
            # 获取初始变换（随机选取 Z > 0.03 的点作为初始值）
            initial_transform = get_random_initial_transformation(remaining_cloud, z_threshold=0.03)

            # 使用 ICP 算法进行点云匹配
            reg = o3d.pipelines.registration.registration_icp(
                remaining_cloud, model, max_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            if reg.fitness < min_fitness:
                print(f"第 {iteration + 1} 次匹配结束：匹配分数 {reg.fitness:.4f} 低于阈值 {min_fitness}")
                continue

            # 提取 ICP 结果的变换矩阵
            transformation = reg.transformation
            rotation_matrix = transformation[:3, :3]
            translation_vector = transformation[:3, 3]

            print(f"第 {iteration + 1} 次匹配成功: 匹配分数 = {reg.fitness:.4f}")
            print(f"位置 (平移向量): {translation_vector}")
            print(f"姿态 (旋转矩阵):\n{rotation_matrix}")

            results.append({
                "instance_id": iteration + 1,
                "fitness_score": reg.fitness,
                "rotation_matrix": rotation_matrix,
                "translation_vector": translation_vector
            })

            # 使用模型的变换结果移除匹配点
            transformed_model = model.transform(transformation)
            distances = remaining_cloud.compute_point_cloud_distance(transformed_model)
            remaining_indices = [i for i, d in enumerate(distances) if d > max_distance]
            remaining_cloud = remaining_cloud.select_by_index(remaining_indices)

            if len(remaining_cloud.points) == 0:
                print("剩余点云为空，匹配结束")
                break
        except Exception as e:
            print(f"匹配过程出错: {e}")
            break

    return results

def match_point_cloud_with_models(pointcloud, models, max_distance, min_fitness=0.3):
    """
    匹配点云与多个模型，包括场景中同一模型的多个实例
    :param pointcloud: 待匹配的点云
    :param models: 多个模型的点云列表
    :param max_distance: ICP 的最大对应点距离
    :param min_fitness: 最低匹配分数阈值，低于此值不认为是匹配
    :param max_iterations: 最大迭代次数（每个模型的实例数量上限）
    :return: 匹配结果列表，每个结果包含模型名称、实例 ID、平移向量、旋转矩阵和匹配分数
    """
    all_results = []
    
    for model_idx, model in enumerate(models):
        print(f"\n开始匹配模型 {model_idx} ({len(model.points)} 点)")
        model_results = match_multiple_instances(
            pointcloud, model, max_distance, min_fitness
        )
        
        # 为每个匹配结果添加模型信息
        for result in model_results:
            result["model_name"] = f"Model_{model_idx}"
            all_results.append(result)
    
    return all_results

def visualize_scene(pointcloud, models, coordinate_frame_size=0.5, origin=[0, 0, 0]):
    """
    可视化点云、模型和坐标系
    :param pointcloud: 原始点云
    :param models: 所有加载的 OBJ 模型
    :param coordinate_frame_size: 坐标系的大小（默认 1.0）
    :param origin: 坐标系的原点位置，默认为 [0, 0, 0]
    """
    geometries = [pointcloud]
    geometries.extend(models)

    # 创建并添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coordinate_frame_size,
        origin=origin
    )
    geometries.append(coordinate_frame)

    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud and OBJ Models with Coordinate Frame",
        width=800,
        height=600
    )

def visualize_scenes(pointclouds, coordinate_frame_size=0.5, origin=[0, 0, 0]):
    """
    可视化多个点云场景
    :param pointclouds: 点云列表，每个元素是一个 Open3D 点云对象
    :param coordinate_frame_size: 坐标系的大小（默认 0.5）
    :param origin: 坐标系的原点位置，默认为 [0, 0, 0]
    """
    if not pointclouds:
        print("未提供任何点云进行可视化")
        return
    
    # 准备几何对象列表
    geometries = []
    
    # 添加所有点云到几何列表
    for i, pointcloud in enumerate(pointclouds):
        if isinstance(pointcloud, o3d.geometry.PointCloud):
            geometries.append(pointcloud)
            print(f"已添加点云 {i} 进行可视化，包含 {len(pointcloud.points)} 个点")
        else:
            print(f"点云 {i} 不是有效的 Open3D 点云对象，已跳过")
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coordinate_frame_size,
        origin=origin
    )
    geometries.append(coordinate_frame)
    
    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Multiple Point Clouds Visualization",
        width=1200,
        height=800
    )

def visualize_models(models):
    """
    仅可视化加载的模型
    :param models: 模型点云列表
    """
    if not models:
        print("未提供任何模型进行可视化")
        return
    
    o3d.visualization.draw_geometries(
        models, 
        window_name="OBJ Models Visualization",
        width=800, height=600
    )

def visualize_results(pointcloud, results, coordinate_frame_size=0.2, origin=[0, 0, 0]):
    """
    可视化点云和匹配结果，包括变换后的模型实例坐标系
    :param pointcloud: 原始点云
    :param results: 匹配结果列表，每个包含平移向量和旋转矩阵
    :param coordinate_frame_size: 坐标系的大小
    """
    geometries = [pointcloud]
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coordinate_frame_size,
        origin=origin
    )
    geometries.append(coordinate_frame)

    # 添加每个匹配实例的坐标系
    for result in results:
        translation = result['translation_vector']
        rotation = result['rotation_matrix']

        # 创建坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coordinate_frame_size
        )
        # 应用旋转和平移
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        coordinate_frame.transform(transformation)
        geometries.append(coordinate_frame)
    
    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud with Results",
        width=800,
        height=600
    )

def move_models(models, translation):
    """
    移动模型列表中的所有模型
    :param models: 模型列表，包含多个 Open3D 的 TriangleMesh 对象
    :param translation: 平移向量，例如 [x, y, z]
    :return: 移动后的模型列表
    """
    moved_models = []
    for i, model in enumerate(models):
        # 克隆模型（防止修改原模型）
        moved_model = model.translate(translation, relative=True)
        moved_models.append(moved_model)
        print(f"模型 {i} 已移动，平移向量: {translation}")
    return moved_models

# 使用Fast Global Registration (FGR)
def global_registration(cube, pointcloud):
    # 创建特征提取器，使用FPFH特征
    voxel_size = 0.01  # 调整合适的体素大小
    cube_down = cube.voxel_down_sample(voxel_size)
    pointcloud_down = pointcloud.voxel_down_sample(voxel_size)

    # 估计法线
    pointcloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 计算FPFH特征
    pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pointcloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
        
    cube_fpfh = o3d.pipelines.registration.compute_fpfh_feature(cube_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    pointcloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pointcloud_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    
    # 使用FGR进行全局配准
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        cube_down, pointcloud_down, cube_fpfh, pointcloud_fpfh, 
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=0.001,
            use_absolute_scale=False,  # 默认值
            decrease_mu=False,  # 默认值
            iteration_number=64,  # 默认值
            maximum_tuple_count=10000  # 默认值
        )
    )
    
    return result

def main():
    # 配置路径
    # pointcloud_path = "/opt/ros_ws/tmp/zed_point_cloud.ply"  # 替换为实际点云路径
    pointcloud_path = "/opt/ros_ws/tmp/zed_point_cloud_world.ply"  # 替换为实际点云路径
    pointcloud_path2 = "/opt/ros_ws/tmp/zed_point_cloud_world2.ply"  # 替换为实际点云路径
    obj_dir = "/opt/ros_ws/src/franka_zed_gazebo/meshes"  # 替换为 OBJ 模型路径

    # 加载点云
    pointcloud = load_point_cloud(pointcloud_path)

    # 使用 Open3D 加载 OBJ 模型并添加距离
    models = load_obj_models(obj_dir)
    
    # 匹配点云和模型
    print("开始匹配点云和模型...")
    cube = models[0]
    # results = match_point_cloud_with_models(pointcloud, models, 0.01)
    # print("\n匹配结果:")

    result = global_registration(cube, pointcloud)

    # 查看配准结果
    print("Transformation Matrix:\n", result.transformation)
    cube.transform(result.transformation)

    # 可视化
    o3d.visualization.draw_geometries([cube, pointcloud])

    # 打印结果
    # for result in results:
    #     print(f"\n模型: {result['model_name']} - 实例 ID: {result['instance_id']}")
    #     print(f"匹配分数: {result['fitness_score']:.4f}")
    #     print(f"位置 (平移向量): {result['translation_vector']}")
    #     print(f"姿态 (旋转矩阵):\n{result['rotation_matrix']}")
    # visualize_results(pointcloud, results)

    # models = move_models(models,  [0.36796876, 0.17114625, 0.03769435])
    # pointcloud = load_point_cloud(pointcloud_path)
    # visualize_results(pointcloud, results)

    # pointcloud2 = load_point_cloud(pointcloud_path2)
    # 可视化点云和模型
    # visualize_scene(pointcloud, models)
    # visualize_scene(pointcloud2, models)
    # visualize_scenes([pointcloud,pointcloud2])
    # visualize_models(models)

if __name__ == "__main__":
    main()
