import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from cv_bridge import CvBridge
import os
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import numpy as np
import cv2
import struct

def parse_rgb_float(rgb_float):
    # 将float32编码的rgb值转换为整数
    rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
    # 按位提取rgb值
    red = (rgb_int >> 16) & 0x0000ff
    green = (rgb_int >> 8) & 0x0000ff
    blue = (rgb_int) & 0x0000ff
    return (red, green, blue)

class PointCloudSaver:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('point_cloud_saver', anonymous=True)
        self.bridge = CvBridge()
        self.camera_info = None
        self.point_cloud_data = None  # Store the point cloud data here
        self.color_image = None       # Store the color image here

        # TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to topics
        rospy.Subscriber("/zed2/zed_node/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)
        rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)  # Image callback
        rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, self.camera_info_callback)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg

    def point_cloud_callback(self, msg):
        if self.point_cloud_data is None:
            self.point_cloud_data = msg
            rospy.loginfo("Received point cloud data.")

    def image_callback(self, msg):
        if self.color_image is None:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rospy.loginfo("Received image message.")

    def get_transform(self, from_frame, to_frame):
        rospy.loginfo(f"Requesting transform from {from_frame} to {to_frame}...")
        transform = self.tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time(0), rospy.Duration(1.0))
        rospy.loginfo(f"Transform found: {transform}")
        return transform

    def save_point_clouds(self, world_path):
        while self.point_cloud_data is None or self.color_image is None:
            rospy.sleep(0.1)

        # Process the point cloud here, when saving
        point_list = pc2.read_points(self.point_cloud_data, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        points = []
        colors = []

        for point in point_list:
            if point[2] > 1.5:
                continue
            points.append([point[0], point[1], point[2]])  # X, Y, Z
            r, g, b = parse_rgb_float(point[3])
            colors.append([b / 255.0, g / 255.0 , r / 255.0])


        if len(points) == 0:
            rospy.logwarn("Received an empty point cloud!")
            return

        # 添加颜色验证
        colors_array = np.array(colors)
        rospy.loginfo(f"Color range - Min: {np.min(colors_array, axis=0)}, Max: {np.max(colors_array, axis=0)}")

        # Get transform
        # TODO: point cloud right 
        transform = self.get_transform('world', 'left_camera_link_optical')
        if transform is None:
            rospy.logerr("Failed to get transform to world frame!")
            return

        # Transform point cloud to world coordinate frame
        transformed_point_cloud = self.transform_point_cloud(points, colors, transform)

        # Debugging: check if the point cloud is empty or not
        # num_points = len(np.asarray(transformed_point_cloud.points))
        # rospy.loginfo(f"Transformed point cloud contains {num_points} points.")

        # if num_points == 0:
            # rospy.logwarn("The transformed point cloud is empty.")
        # else:
            # Visualize the point cloud using Open3D
            # o3d.visualization.draw_geometries([transformed_point_cloud])
            # pass
        # o3d.visualization.draw_geometries([transformed_point_cloud])

        # Save the transformed point cloud
        o3d.io.write_point_cloud(world_path, transformed_point_cloud)
        rospy.loginfo(f"Transformed point cloud saved to {world_path}")

    def project_to_image(self, point, K):
        # 将3D点投影到2D图像坐标系
        X, Y, Z = point
        uvw = np.dot(K, np.array([X, Y, Z]))
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        return u, v

    def transform_point_cloud(self, points, colors, transform):
        translation = np.array([transform.transform.translation.x,
                                 transform.transform.translation.y,
                                 transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.w,
                             transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z])

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
        transformation_matrix[:3, 3] = translation

        # Create Open3D point cloud from processed points and colors
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

        # Apply transform to the point cloud
        point_cloud.transform(transformation_matrix)
        return point_cloud


if __name__ == "__main__":
    point_cloud_saver = PointCloudSaver()

    # Wait for data to be ready

    # Save and display point clouds
    world_file = "/opt/ros_ws/src/franka_zed_gazebo/scripts/mycode/2_perception/mesh/zed_point_cloud_world3.ply"
    point_cloud_saver.save_point_clouds(world_file)
