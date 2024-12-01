import rospy
import tf2_ros
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import csv
from tf2_geometry_msgs import do_transform_pose


class DepthColorImageSaver:
    def __init__(self):
        rospy.init_node('depth_color_image_saver', anonymous=True)

        # 订阅深度图像和彩色图像
        rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, self.save_depth_image)
        rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.save_color_image)
        
        # 订阅 tf 信息
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.bridge = CvBridge()
        
        # 保存路径
        self.depth_save_dir = "/opt/ros_ws/tmp/depths"
        self.color_save_dir = "/opt/ros_ws/tmp/colors"
        self.csv_file = "/opt/ros_ws/tmp/image_tf_record.csv"

        # 创建目录
        if not os.path.exists(self.depth_save_dir):
            os.makedirs(self.depth_save_dir)
        if not os.path.exists(self.color_save_dir):
            os.makedirs(self.color_save_dir)

        # CSV 文件初始化
        self.csv_initialized = False

    def save_depth_image(self, msg):
        try:
            # 获取TF转换以获得时间戳
            tf_data = self.get_transform('world', 'left_camera_link')
            if tf_data is None:
                return

            timestamp = tf_data.header.stamp  # 使用TF的时间戳

            # 将深度图像转换为 OpenCV 格式
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # 确保深度图是 16 位灰度图或转换为 16 位
            if depth_image.dtype == np.float32:
                depth_image = (depth_image * 1000).astype(np.uint16)  # 转换为毫米单位

            # 使用时间戳生成文件名
            depth_filename = f"depth_{timestamp.to_nsec()}.png"
            depth_filepath = os.path.join(self.depth_save_dir, depth_filename)

            # 如果文件已存在，替换文件
            if os.path.exists(depth_filepath):
                os.remove(depth_filepath)

            # 保存图像
            cv2.imwrite(depth_filepath, depth_image)
            rospy.loginfo(f"Depth image saved to {depth_filepath}")

            # 保存TF信息到CSV
            self.save_tf_to_csv(depth_filename, tf_data)

        except Exception as e:
            rospy.logerr(f"Failed to save depth image: {e}")

    def save_color_image(self, msg):
        try:
            # 获取TF转换以获得时间戳
            tf_data = self.get_transform('world', 'left_camera_link')
            if tf_data is None:
                return

            timestamp = tf_data.header.stamp  # 使用TF的时间戳

            # 将彩色图像转换为 OpenCV 格式
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 使用时间戳生成文件名
            color_filename = f"color_{timestamp.to_nsec()}.png"
            color_filepath = os.path.join(self.color_save_dir, color_filename)

            # 如果文件已存在，替换文件
            if os.path.exists(color_filepath):
                os.remove(color_filepath)

            # 保存图像
            cv2.imwrite(color_filepath, color_image)
            rospy.loginfo(f"Color image saved to {color_filepath}")

            # 保存TF信息到CSV
            self.save_tf_to_csv(color_filename, tf_data)

        except Exception as e:
            rospy.logerr(f"Failed to save color image: {e}")

    def get_transform(self, from_frame, to_frame):
        try:
            # 获取从from_frame到to_frame的变换
            rospy.loginfo(f"Requesting transform from {from_frame} to {to_frame}...")

            # 查询变换
            transform = self.tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time(0), rospy.Duration(1.0))  # 设置超时为1秒
            
            rospy.loginfo(f"Transform found: {transform}")

            # 返回TransformStamped对象
            return transform

        except (tf2_ros.TransformException) as e:
            rospy.logerr(f"Could not get transform from {from_frame} to {to_frame}: {e}")
            return None

    def save_tf_to_csv(self, filename, tf_data):
        try:
            # 打开CSV文件
            if not self.csv_initialized:
                with open(self.csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # 写入表头
                    writer.writerow(['Image Filename', 'TF Translation X', 'TF Translation Y', 'TF Translation Z', 'TF Rotation X', 'TF Rotation Y', 'TF Rotation Z', 'TF Rotation W'])

                self.csv_initialized = True

            # 获取TF转换信息
            translation = tf_data.transform.translation
            rotation = tf_data.transform.rotation

            # 将数据写入CSV文件
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename, translation.x, translation.y, translation.z, rotation.x, rotation.y, rotation.z, rotation.w])

            rospy.loginfo(f"TF data saved for {filename} in CSV file.")

        except Exception as e:
            rospy.logerr(f"Failed to save TF data to CSV: {e}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    saver = DepthColorImageSaver()
    saver.run()
