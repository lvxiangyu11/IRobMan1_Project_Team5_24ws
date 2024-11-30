#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import tf2_ros
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class PointCloudSaver:
    def __init__(self):
        rospy.init_node('point_cloud_saver', anonymous=True)

        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_image = None
        self.color_image = None

        # TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to topics
        rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback)
        rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
        rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, self.camera_info_callback)

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Error converting depth image: {e}")

    def image_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting color image: {e}")

    def get_transform(self, from_frame, to_frame):
        """
        获取从 from_frame 到 to_frame 的变换
        """
        try:
            rospy.loginfo(f"Requesting transform from {from_frame} to {to_frame}...")
            transform = self.tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time(0), rospy.Duration(1.0))
            rospy.loginfo(f"Transform found: {transform}")
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Could not get transform from {from_frame} to {to_frame}: {e}")
            return None

    def generate_point_cloud(self):
        """
        根据深度图和RGB图生成点云
        """
        if self.depth_image is None or self.color_image is None or self.camera_info is None:
            rospy.logwarn("Waiting for depth image, color image, and camera info...")
            return None

        # 获取相机内参
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # 创建点云
        points = []
        colors = []

        for v in range(self.depth_image.shape[0]):
            for u in range(self.depth_image.shape[1]):
                z = self.depth_image[v, u] / 1.0  # rescale
                if z > 0:  # 过滤无效的深度值
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])

                    # 获取像素颜色
                    b, g, r = self.color_image[v, u]
                    colors.append([r / 255.0, g / 255.0, b / 255.0])

        points = np.array(points)
        colors = np.array(colors)

        # 创建Open3D点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud

    def transform_point_cloud(self, point_cloud, transform):
        """
        使用TF变换将点云转换到世界坐标系
        """
        # 从TransformStamped提取变换矩阵
        translation = np.array([transform.transform.translation.x,
                                 transform.transform.translation.y,
                                 transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.w,
                             transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             ])

        # 转换为4x4变换矩阵
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
        transformation_matrix[:3, 3] = translation

        # 应用变换到点云
        point_cloud.transform(transformation_matrix)
        return point_cloud

    def save_point_clouds(self, original_path, world_path):
        """
        保存原始点云和转换到世界坐标系的点云
        """
        # 生成点云
        point_cloud = self.generate_point_cloud()
        if point_cloud is None:
            rospy.logerr("Failed to generate point cloud!")
            return

        # 保存原始点云
        o3d.io.write_point_cloud(original_path, point_cloud)
        rospy.loginfo(f"Original point cloud saved to {original_path}")

        # 获取变换
        transform = self.get_transform('world', 'left_camera_link_optical')
        if transform is None:
            rospy.logerr("Failed to get transform to world frame!")
            return

        # 转换点云到世界坐标系并保存
        transformed_point_cloud = self.transform_point_cloud(point_cloud, transform)
        o3d.io.write_point_cloud(world_path, transformed_point_cloud)
        rospy.loginfo(f"Transformed point cloud saved to {world_path}")


if __name__ == "__main__":
    point_cloud_saver = PointCloudSaver()

    # 等待数据准备
    rospy.loginfo("Waiting for data...")
    rospy.sleep(2)  # 等待话题数据发布

    # 保存点云
    original_file = "/opt/ros_ws/tmp/zed_point_cloud3.ply"
    world_file = "/opt/ros_ws/tmp/zed_point_cloud_world3.ply"
    point_cloud_saver.save_point_clouds(original_file, world_file)
