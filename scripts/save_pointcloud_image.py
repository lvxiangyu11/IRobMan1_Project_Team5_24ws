import rospy
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import cv2
from cv_bridge import CvBridge
import os

class DataSaver:
    def __init__(self):
        rospy.init_node('data_saver', anonymous=True)

        # 订阅点云和图像
        rospy.Subscriber('/zed2/zed_node/depth/depth_registered', PointCloud2, self.save_pointcloud)
        rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.save_image)

        self.bridge = CvBridge()
        self.image_saved = False
        self.pointcloud_saved = False

        # 确保目录存在
        self.save_dir = "/opt/ros_ws/tmp"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_pointcloud(self, msg):
        if not self.pointcloud_saved:
            # 解析点云数据
            points = []
            for point in pc2.read_points(msg, skip_nans=True):
                points.append([point[0], point[1], point[2]])

            # 使用 Open3D 保存点云
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            filepath = os.path.join(self.save_dir, 'pointcloud.pcd')
            o3d.io.write_point_cloud(filepath, cloud)
            rospy.loginfo(f"PointCloud saved to {filepath}")

            self.pointcloud_saved = True

    def save_image(self, msg):
        if not self.image_saved:
            # 转换 ROS 图像为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 保存图像为 PNG
            filepath = os.path.join(self.save_dir, 'image.png')
            cv2.imwrite(filepath, cv_image)
            rospy.loginfo(f"Image saved to {filepath}")

            self.image_saved = True

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    saver = DataSaver()
    saver.run()
