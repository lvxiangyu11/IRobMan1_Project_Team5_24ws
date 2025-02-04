import numpy as np
import open3d as o3d
import numpy as np
from scipy.ndimage import gaussian_filter

def max_downsample_image(image, pool_size=(2, 2)):
    """
    Downsample the image using max pooling.

    Args:
        image (numpy.ndarray): Input color image (height, width, 3)
        pool_size (tuple): Pooling window size (height, width)

    Returns:
        downsampled_image (numpy.ndarray): Downsampled image
    """
    h, w, c = image.shape
    ph, pw = pool_size

    # Ensure the image size is a multiple of the pooling window size
    h_new = h // ph * ph
    w_new = w // pw * pw
    image_cropped = image[:h_new, :w_new, :]

    # Perform max pooling for downsampling
    downsampled_image = image_cropped.reshape(h_new // ph, ph, w_new // pw, pw, c).max(axis=(1, 3))

    return downsampled_image

def pointcloud_to_top_view_image_color(pointcloud, voxel_size=0.01, output_image_size=(512, 512)):
    """
    Generate a top-down 2D color image from the point cloud.
    
    Args:
        pointcloud (o3d.geometry.PointCloud): Input point cloud
        voxel_size (float): Resolution for dividing the grid
        output_image_size (tuple): Output image size (width, height)
    
    Returns:
        top_view_image (numpy.ndarray): Generated 2D color image
    """
    # Get the point coordinates and colors from the point cloud
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else np.zeros_like(points)

    # Extract the x-y range
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    print(x_max, y_max)

    # Define grid resolution
    width, height = output_image_size
    grid_x = np.linspace(x_min, x_max, width)
    grid_y = np.linspace(y_min, y_max, height)

    # Initialize the 2D color image
    top_view_image = np.zeros((height, width, 3))  # RGB image

    # Initialize the count matrix to accumulate colors
    count_matrix = np.zeros((height, width))

    # Iterate over the point cloud and project each point onto the grid
    for point, color in zip(points, colors):
        x, y, z = point
        # Find the corresponding grid index
        x_idx = int((x - x_min) / (x_max - x_min) * (width - 1))
        y_idx = int((y - y_min) / (y_max - y_min) * (height - 1))
        
        # Update the color image and count matrix
        top_view_image[height - 1 - y_idx, x_idx] += color  # Accumulate color
        count_matrix[height - 1 - y_idx, x_idx] += 1

    # Average the color values
    nonzero_mask = count_matrix > 0
    top_view_image[nonzero_mask] /= count_matrix[nonzero_mask, None]

    # Normalize the result to the range [0, 1]
    top_view_image = np.clip(top_view_image, 0, 1)

    return top_view_image


import scipy.ndimage

def interpolate_sparse_image(image, dilation_size=(7, 7)):
    """
    Perform interpolation filling for sparse color images.
    
    Args:
        image (numpy.ndarray): Input color image (height, width, 3)
        dilation_size (tuple): Neighborhood size for dilation operation (height, width)
    
    Returns:
        interpolated_image (numpy.ndarray): Interpolated color image
    """
    interpolated_image = image.copy()
    for i in range(3):  # Handle RGB channels separately
        interpolated_image[:, :, i] = scipy.ndimage.morphology.grey_dilation(
            image[:, :, i], size=dilation_size
        )
    return interpolated_image


def pointcloud_to_colored_image_with_filling(pointcloud, output_size=(512, 512), voxel_size=0.01):
    """
    Generate a top-down 2D color image from the point cloud and fill blank areas using Gaussian filtering.
    
    Args:
        pointcloud (o3d.geometry.PointCloud): Input point cloud.
        output_size (tuple): Output image size (width, height).
        voxel_size (float): Grid resolution for dividing the grid.
    
    Returns:
        top_view_image (numpy.ndarray): Filled color image.
    """
    # Get the point coordinates and colors from the point cloud
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else np.zeros_like(points)

    # Extract the x-y range
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    # Define grid resolution
    width, height = output_size
    grid_x = np.linspace(x_min, x_max, width)
    grid_y = np.linspace(y_min, y_max, height)

    # Initialize the 2D color image and count matrix
    top_view_image = np.zeros((height, width, 3), dtype=np.float32)  # RGB image
    count_matrix = np.zeros((height, width), dtype=np.float32)

    # Iterate over the point cloud and project each point onto the grid
    for point, color in zip(points, colors):
        x, y, z = point
        # Find the corresponding grid index
        x_idx = int((x - x_min) / (x_max - x_min) * (width - 1))
        y_idx = int((y - y_min) / (y_max - y_min) * (height - 1))
        
        # Accumulate color values
        top_view_image[height - 1 - y_idx, x_idx] += color  # Accumulate color
        count_matrix[height - 1 - y_idx, x_idx] += 1

    # Normalize color values
    nonzero_mask = count_matrix > 0
    top_view_image[nonzero_mask] /= count_matrix[nonzero_mask, None]

    # Fill blank areas using Gaussian filtering
    for channel in range(3):  # Process R/G/B channels separately
        top_view_image[:, :, channel] = gaussian_filter(top_view_image[:, :, channel], sigma=2)

    # Clip the image to [0, 1] range
    top_view_image = np.clip(top_view_image, 0, 1)

    return top_view_image

def triangle_mesh_to_image(mesh, image_size=(100, 100), fill_radius=1):
    """
    Project the vertices of a TriangleMesh onto a 2D image and generate a smaller image using max pooling.
    Ensure that each pixel has a value.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input triangle mesh.
        image_size (tuple): Output image size (width, height).
        pool_size (tuple): Pooling window size (height, width).
        fill_radius (int): Neighborhood radius (in pixels) for each point's influence.
    
    Returns:
        numpy.ndarray: 2D image after projection (size after pooling).
    """
    # Get vertices and colors
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones((len(vertices), 3))

    # Extract the x-y range
    x_min, y_min = vertices[:, 0].min(), vertices[:, 1].min()
    x_max, y_max = vertices[:, 0].max(), vertices[:, 1].max()

    # Initialize the 2D image
    width, height = image_size
    image = np.zeros((height, width, 3), dtype=np.float32)
    count_matrix = np.zeros((height, width), dtype=np.int32)  # For counting pixel value accumulation

    # Map the vertices to image coordinates and fill the surrounding pixels
    for vertex, color in zip(vertices, colors):
        x, y = vertex[0], vertex[1]
        u = int((x - x_min) / (x_max - x_min) * (width - 1))
        v = int((y - y_min) / (y_max - y_min) * (height - 1))

        # Fill the current point and the surrounding pixels
        for du in range(-fill_radius, fill_radius + 1):
            for dv in range(-fill_radius, fill_radius + 1):
                uu = min(max(u + du, 0), width - 1)
                vv = min(max(v + dv, 0), height - 1)
                image[vv, uu] += color
                count_matrix[vv, uu] += 1

    # Average the color values for each pixel
    mask = count_matrix > 0
    image[mask] /= count_matrix[mask, None]

    # Convert the image to the range 0-255
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    return image

def enlarge_points_as_cubes(point_cloud, cube_size=0.0001):
    """
    Replace each point in the point cloud with a cube to enlarge the display size.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        cube_size (float): Cube edge length.
    
    Returns:
        o3d.geometry.TriangleMesh: A combined geometry of all cubes representing the points.
    """
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else [[1, 0, 0]] * len(points)
    mesh = o3d.geometry.TriangleMesh()

    for point, color in zip(points, colors):
        # Create a small cube and move it to the point's position
        cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
        cube.translate(point - np.array([cube_size / 2, cube_size / 2, cube_size / 2]))
        cube.paint_uniform_color(color)
        mesh += cube  # Add the cube to the combined mesh

    return mesh
