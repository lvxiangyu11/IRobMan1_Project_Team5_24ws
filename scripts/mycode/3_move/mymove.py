#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler, quaternion_multiply, quaternion_inverse
import numpy as np

class MoveRobot:
    def __init__(self):
        # 初始化 MoveIt 和 ROS 节点
        moveit_commander.roscpp_initialize(sys.argv)
        if not rospy.core.is_initialized():
            rospy.init_node("my_gripper_node", anonymous=True)
        # 初始化机器人接口
        self.robot = moveit_commander.RobotCommander()
        self.group_name = "panda_arm"  # 根据你的机器人调整
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        rospy.loginfo("MoveRobot initialized successfully.")

    def move(self, position, rpy):
        """根据目标位置和姿态移动机器人"""
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

    def grasp_approach(self, start_position, end_position, rpy):
        """
        从起始位置逐步接近目标位置，同时保持爪子的方向与目标姿态一致。
        """
        try:
            # 将 RPY 转换为四元数
            quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            # 生成路径点列表
            waypoints = []
            step_count = 2  # 插值步数

            for i in range(step_count + 1):
                t = i / float(step_count)
                pose = geometry_msgs.msg.Pose()
                pose.position.x = start_position[0] * (1 - t) + end_position[0] * t
                pose.position.y = start_position[1] * (1 - t) + end_position[1] * t
                pose.position.z = start_position[2] * (1 - t) + end_position[2] * t
                pose.orientation.x = quaternion[0]
                pose.orientation.y = quaternion[1]
                pose.orientation.z = quaternion[2]
                pose.orientation.w = quaternion[3]
                waypoints.append(pose)

            # 打印路径点调试信息
            rospy.loginfo("Generated waypoints:")
            for idx, wp in enumerate(waypoints):
                rospy.loginfo(
                    f"Waypoint {idx}: Position({wp.position.x}, {wp.position.y}, {wp.position.z}), "
                    f"Orientation({wp.orientation.x}, {wp.orientation.y}, {wp.orientation.z}, {wp.orientation.w})"
                )

            # 笛卡尔路径规划
            rospy.loginfo("Planning Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,   # 路径点
                0.01,        # 最大步长
                True,        # 避免碰撞
                0.0          # 跳跃阈值
            )

            # 检查路径规划结果
            if fraction < 1.0:
                rospy.logwarn(f"Path planning succeeded for only {fraction * 100:.2f}% of the path")
                return False

            rospy.loginfo("Path planning completed successfully!")

            # 执行路径
            rospy.loginfo("Executing Cartesian path...")
            success = self.move_group.execute(plan, wait=True)

            # 停止和清理
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            if success:
                rospy.loginfo("Grasp approach executed successfully.")
            else:
                rospy.logwarn("Grasp approach execution failed.")

            return success

        except Exception as e:
            rospy.logerr(f"Exception in grasp_approach: {e}")
            return False



    def __del__(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveRobot shut down.")


if __name__ == "__main__":
    try:
        robot_mover = MoveRobot()

        # 初始和目标位置
        start_position = [0.4, 0, 0.5]
        end_position = [0.4, 0, 0.14+0.1]
        target_rpy = [0, np.pi, np.pi/4]

        rospy.loginfo("Starting grasp approach...")
        robot_mover.grasp_approach(start_position, end_position, target_rpy)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
