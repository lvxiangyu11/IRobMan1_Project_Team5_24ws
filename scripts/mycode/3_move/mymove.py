#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf.transformations import quaternion_from_euler
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject


class MoveRobotWithCollision:
    def __init__(self):
        # 初始化 MoveIt 和 ROS 节点
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_robot_with_collision_node", anonymous=True)

        # 初始化机器人接口
        self.robot = moveit_commander.RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "panda_arm"  # 根据你的机器人调整
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # 初始化 TF 变换监听器
        self.tf_listener = tf.TransformListener()

        # 初始化深度图桥接器
        self.bridge = CvBridge()

        # 设置点云订阅
        self.depth_topic = "/zed2/zed_node/depth/depth_registered"
        self.depth_subscriber = rospy.Subscriber(
            self.depth_topic, Image, self.depth_callback
        )
        self.latest_depth = None

        rospy.loginfo("MoveRobotWithCollision initialized successfully.")

    def depth_callback(self, depth_msg):
        """接收深度图数据的回调函数"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            rospy.loginfo("Depth data received successfully.")
        except Exception as e:
            rospy.logerr("Failed to convert depth image: %s", str(e))

    def get_transform(self, target_frame, source_frame):
        """获取指定帧之间的变换"""
        try:
            self.tf_listener.waitForTransform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except tf.Exception as e:
            rospy.logerr("TF lookup failed: %s", str(e))
            return None, None

    def update_collision_objects(self):
        """基于深度图更新碰撞对象"""
        if self.latest_depth is None:
            rospy.logwarn("No depth data received yet. Skipping collision update.")
            return

        # 获取相机到全局坐标的变换
        trans, rot = self.get_transform("world", "left_camera_link")
        if trans is None or rot is None:
            rospy.logwarn("Failed to get TF transform. Skipping collision update.")
            return

        # 将深度图转化为点云
        height, width = self.latest_depth.shape
        fx, fy = 525.0, 525.0  # 相机焦距，根据你的相机设置调整
        cx, cy = width / 2, height / 2

        points = []
        for v in range(0, height, 10):  # 减少计算量，每 10 行采样一次
            for u in range(0, width, 10):  # 每 10 列采样一次
                z = self.latest_depth[v, u]
                if np.isnan(z) or z <= 0:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

        # 转换到全局坐标
        points_global = []
        for point in points:
            p = np.array(point)
            rotation_matrix = tf.transformations.quaternion_matrix(rot)[:3, :3]
            point_global = np.dot(rotation_matrix, p) + np.array(trans)
            points_global.append(point_global)

        # 添加碰撞对象到规划场景
        self.scene.remove_world_object("obstacle")  # 先清除之前的障碍物
        co = CollisionObject()
        co.id = "obstacle"
        co.header.frame_id = "world"

        # 使用点云的包围盒作为碰撞对象
        points_global = np.array(points_global)
        if points_global.size > 0:
            min_point = np.min(points_global, axis=0)
            max_point = np.max(points_global, axis=0)
            size = max_point - min_point

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [size[0], size[1], size[2]]

            co.primitives.append(primitive)
            co.primitive_poses.append(
                geometry_msgs.msg.Pose(
                    position=geometry_msgs.msg.Point(
                        x=(min_point[0] + max_point[0]) / 2,
                        y=(min_point[1] + max_point[1]) / 2,
                        z=(min_point[2] + max_point[2]) / 2,
                    ),
                    orientation=geometry_msgs.msg.Quaternion(0, 0, 0, 1),
                )
            )
            co.operation = CollisionObject.ADD
            self.scene.add_object(co)
            rospy.loginfo("Updated collision object based on depth data.")
        else:
            rospy.logwarn("No valid points found in depth data. Skipping collision update.")

    def move(self, position, rpy):
        """根据目标位置和姿态移动机器人"""
        self.update_collision_objects()

        # 将 RPY 转换为四元数
        quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = position[0]
        pose_goal.position.y = position[1]
        pose_goal.position.z = position[2]

        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]

        self.move_group.set_pose_target(pose_goal)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            rospy.loginfo("Move successful to position: {} and RPY: {}".format(position, rpy))
        else:
            rospy.logwarn("Move failed. Check collision or planning constraints.")

        return success

    def __del__(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveRobotWithCollision shut down.")


if __name__ == "__main__":
    try:
        robot_mover = MoveRobotWithCollision()
        target_position = [0.4, 0, 0.13 + 0.0]  # 目标位置 (x, y, z)
        target_rpy = [0, np.pi, np.pi/2+np.pi/4]  # 目标姿态 (roll, pitch, yaw)
        rospy.loginfo("Starting move with collision avoidance...")
        robot_mover.move(target_position, target_rpy)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
