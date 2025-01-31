#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler
import numpy as np


class MoveRobot:
    def __init__(self):
        # 初始化 MoveIt 和 ROS 节点
        moveit_commander.roscpp_initialize(sys.argv)
        if not rospy.core.is_initialized():
            rospy.init_node("my_gripper_node", anonymous=True)
        # 初始化机器人接口
        self.robot = moveit_commander.RobotCommander()
        self.group_name = "panda_manipulator"  # 根据你的机器人调整
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        rospy.loginfo("MoveRobot initialized successfully.")

    def add_constraints(self):
        """添加路径约束，限制机器人在 z > 0.01 的区域内活动"""
        current_pose = self.move_group.get_current_pose().pose
        constraints = moveit_commander.Constraints()

        # 设置约束
        constraints.name = "z_above_0.01"
        self.move_group.set_path_constraints(constraints)

        rospy.loginfo("Constraints added: z > 0.01")

    def clear_constraints(self):
        """清除路径约束"""
        self.move_group.clear_path_constraints()
        rospy.loginfo("Path constraints cleared.")

    def move(self, position, rpy):
        """根据目标位置和姿态移动机器人"""
        # 添加路径约束
        self.add_constraints()

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

        # 清除路径约束
        self.clear_constraints()

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
            step_count = 50  # 插值步数

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

    def get_current_pose(self):
        """获取当前爪子的位置和姿态"""
        try:
            current_pose = self.move_group.get_current_pose().pose
            position = current_pose.position
            orientation = current_pose.orientation
            rospy.loginfo("Current pose: Position({:.3f}, {:.3f}, {:.3f}), Orientation({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ))
            return current_pose
        except Exception as e:
            rospy.logerr(f"Failed to get current pose: {e}")
            return None

    def __del__(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveRobot shut down.")


if __name__ == "__main__":
    try:
        robot_mover = MoveRobot()

        # 获取当前爪子位置
        rospy.loginfo("Getting current pose...")
        current_pose = robot_mover.get_current_pose()
        if current_pose:
            rospy.loginfo("Current position and orientation retrieved successfully.")
            print(current_pose)

        # 初始和目标位置
        start_position = [0.4, 0, 0.5]
        # end_position = [0.4, 0, 0.14 + 0.3]
        end_position = [0.7112641868366598, 0.21893331742514316, 0.13104804613096419] 
        # target_rpy = [0, np.pi, np.pi / 4]
        target_rpy = [-3.1369628913905916, 0.003306838666246037, 1.4534627124557509]
        

        rospy.loginfo("Starting grasp approach...")
        robot_mover.move(end_position, target_rpy)
        # robot_mover.grasp_approach(start_position, end_position, target_rpy)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
