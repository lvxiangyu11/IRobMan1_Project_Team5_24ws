import numpy as np
import open3d as o3d
from typing import Any, Sequence, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
import open3d as o3d
import copy

def generate_pascal_triangle_transforms(t, T):
    """
    Generate a list of transformation matrices for a Pascal triangle arrangement.
    Each cube's center position is used as the translation part of the transform matrix.
    """
    level = 1
    total = 1
    while total < t:
        level += 1
        total += level

    transforms = []
    cube_size = 0.045  # 4.5 cm
    spacing_xy = cube_size * 1.25  # Increase spacing to 1.25 times the cube size
    spacing_z = cube_size * 1.05   # Use the same spacing in the vertical direction

    current_pos = 0
    for row in range(level-1, -1, -1):
        for col in range(row + 1):
            if current_pos >= t:
                break

            center_x = 0
            center_y = (col - row/2) * spacing_xy
            center_z = (level - 1 - row) * spacing_z

            # Create local transformation matrix
            local_transform = np.eye(4)
            local_transform[:3, 3] = [center_x, center_y, center_z]

            # Combine local transformation with T transformation
            transform = np.dot(T, local_transform)
            transforms.append(transform)
            current_pos += 1

        if current_pos >= t:
            break

    return transforms

def create_coordinate_frame(size=0.1, transform=None):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if transform is not None:
        frame.transform(transform)
    return frame

def visualize_pascal_triangle(transforms, T):
    # Create a cube centered at the origin
    cube = o3d.geometry.TriangleMesh.create_box(
        width=0.045,
        height=0.045, 
        depth=0.045
    )
    # Move the cube to be centered at the origin
    cube.translate([-0.045/2, -0.045/2, -0.045/2])
    cube.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add world coordinate frame
    world_frame = create_coordinate_frame(size=0.2)
    vis.add_geometry(world_frame)

    # Add T coordinate frame
    t_frame = create_coordinate_frame(size=0.2, transform=T)
    vis.add_geometry(t_frame)

    # Add all cubes and their local coordinate frames
    for transform in transforms:
        # Add cube
        cube_copy = copy.deepcopy(cube)
        cube_copy.transform(transform)
        vis.add_geometry(cube_copy)
        
        # Add local coordinate frame
        local_frame = create_coordinate_frame(size=0.05, transform=transform)
        vis.add_geometry(local_frame)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

    ctr = vis.get_view_control()
    ctr.set_zoom(0.2)  # Adjust zoom to fit larger spacing
    ctr.set_front([-0.8, -0.5, 0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])

    vis.run()
    vis.destroy_window()

    
def filter_point_cloud_by_depth_and_range(point_cloud, depth_threshold=0.005, range=[0.001, -0.8, 1, 1.6]):
    """
    Read a point cloud file and filter out points with depth below the specified threshold 
    and those that are outside the detection range.

    Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud object.
        depth_threshold (float): Depth threshold, points with depth lower than this value will be filtered.
        range (list): Filtering range, includes [x, y, w, h].
        
    Returns:
        o3d.geometry.PointCloud: The filtered point cloud object.
    """

    # Get the point coordinates from the point cloud
    points = np.asarray(point_cloud.points)

    # Filter out points with depth less than the threshold
    filtered_points = points[points[:, 2] >= depth_threshold]

    # Filter points based on the range (x, y, w, h) -> top-left corner and width-height
    x, y, w, h = range
    filtered_points = filtered_points[
        (filtered_points[:, 0] >= x) & (filtered_points[:, 0] <= x + w) &
        (filtered_points[:, 1] >= y) & (filtered_points[:, 1] <= y + h)
    ]

    # Create a new point cloud object
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    # Retain color information (if it exists)
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_colors = colors[points[:, 2] >= depth_threshold]
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud

def calculate_max_layer(filtered_point_cloud, layer_height=0.04):
    """
    Calculate the maximum layer number (MaxLayer) based on the maximum value along the z-axis, 
    using rounding.

    Args:
        filtered_point_cloud (o3d.geometry.PointCloud): The filtered point cloud object.
        layer_height (float): The height of each layer, default value is 0.04.

    Returns:
        int: The maximum layer number (MaxLayer).
    """
    # Extract the maximum z-axis value from the point cloud
    points = np.asarray(filtered_point_cloud.points)
    z_max = np.max(points[:, 2])

    # Calculate the maximum layer number using strict rounding
    max_layer = int(np.floor(z_max / layer_height + 0.5))

    return max_layer


def filter_point_cloud_by_depth(point_cloud, depth_threshold=0.005):
    """
    Read a point cloud file and filter out points with depth below the specified threshold.

    Args:
        ply_path (str): The path to the point cloud file.
        depth_threshold (float): Depth threshold, points with depth lower than this value will be filtered.

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud object.
    """


    # Get the point coordinates from the point cloud
    points = np.asarray(point_cloud.points)

    # Filter out points with depth less than the threshold
    filtered_points = points[points[:, 2] >= depth_threshold]

    # Create a new point cloud object
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    # Retain color information (if it exists)
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_colors = colors[points[:, 2] >= depth_threshold]
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud


def matrix_to_rpy_and_translation(matrix):
    """
    Extract Roll, Pitch, Yaw (RPY) and translation information from a 4x4 transformation matrix 
    and return it as a list.
    
    Parameters:
        matrix: 4x4 transformation matrix (numpy.ndarray)
        
    Returns:
        result: A list containing [roll, pitch, yaw, x, y, z]
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix")
    
    # Extract the rotation matrix and translation
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Create a copy of the rotation matrix to avoid read-only memory issues
    rotation_matrix = np.array(rotation_matrix)
    
    # Convert to RPY angles using scipy
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    
    # Return the result
    return [roll, pitch, yaw], translation.tolist()


def visualize_3d_objs(objs: Sequence[Any]) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Output viz')
    for obj in objs:
        vis.add_geometry(obj)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def create_grasp_mesh(
    center_point: np.ndarray,
    width: float = 0.05,
    height: float = 0.1,
    depth: float = 0.03,
    gripper_distance: float = 0.1,
    gripper_height: float = 0.1,
    color_left: list = [1, 0, 0],  # Red
    color_right: list = [0, 1, 0],  # Green,
    scale: float = 0.3,
    rotation_matrix: Optional[np.ndarray] = None
) -> Sequence[o3d.geometry.TriangleMesh]:
    """
    Creates a mesh representation of a robotic gripper.

    Args:
        center_point: Central position of the gripper in 3D space
        width: Width of each gripper finger
        height: Height of each gripper finger
        depth: Depth of each gripper finger
        gripper_distance: Distance between gripper fingers
        gripper_height: Height of the gripper base
        color_left: RGB color values for left finger [0-1]
        color_right: RGB color values for right finger [0-1]
        scale: Scaling factor for the gripper dimensions
        rotation_matrix: Optional 3x3 rotation matrix for gripper orientation

    Returns:
        list: List of mesh geometries representing the gripper components
    """
    grasp_geometries = []

    # Apply scaling to dimensions
    width *= scale
    height *= scale
    depth *= scale
    gripper_distance *= scale
    gripper_height *= scale

    # Create left finger
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height,
        depth=depth*2
    )
    left_finger.paint_uniform_color(color_left)
    left_finger.translate((-gripper_distance-width/2, 0, 0) + center_point)
    if rotation_matrix is not None:
        left_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(left_finger)

    # Create right finger
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height,
        depth=depth*2
    )
    right_finger.paint_uniform_color(color_right)
    right_finger.translate((gripper_distance, 0, 0) + center_point)
    if rotation_matrix is not None:
        right_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(right_finger)

    coupler = o3d.geometry.TriangleMesh.create_box(
        width=2*gripper_distance + width,
        height=width/2,
        depth=depth
    )
    coupler.paint_uniform_color([0, 0, 1])
    coupler.translate(
        (-gripper_distance-width/2, gripper_height, 0) + center_point)
    if rotation_matrix is not None:
        coupler.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(coupler)

    stick = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height*1.5,
        depth=depth
    )
    stick.paint_uniform_color([0, 0, 1])
    stick.translate((-width/4, gripper_height, 0) + center_point)
    if rotation_matrix is not None:
        stick.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(stick)

    return grasp_geometries
