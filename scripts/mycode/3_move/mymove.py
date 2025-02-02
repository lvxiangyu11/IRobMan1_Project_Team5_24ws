#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler
import numpy as np
import moveit_msgs.msg
import shape_msgs.msg

class MoveRobot:
    def __init__(self):
        # 初始化 MoveIt 和 ROS 节点
        moveit_commander.roscpp_initialize(sys.argv)
        if not rospy.core.is_initialized():
            rospy.init_node("my_gripper_node", anonymous=True)

        # 等待 move_group action server
        rospy.loginfo("Waiting for move_group action server...")
        try:
            # 初始化机器人接口
            self.robot = moveit_commander.RobotCommander()
            self.group_name = "panda_manipulator"  # 根据你的机器人调整
            self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
            self.add_constraints()
            rospy.loginfo("MoveRobot initialized successfully.")
        except Exception as e:
            rospy.logerr(f"Error initializing MoveRobot: {e}")
            raise

    def add_constraints(self):
        """添加路径约束，限制机器人在 z > 0.001 的区域内活动"""
        constraints = moveit_commander.Constraints()

        # 创建位置约束
        position_constraint = moveit_msgs.msg.PositionConstraint()
        position_constraint.header.frame_id = self.move_group.get_planning_frame()
        position_constraint.link_name = self.move_group.get_end_effector_link()

        # 定义位置约束的区域
        constraint_region = shape_msgs.msg.SolidPrimitive()
        constraint_region.type = shape_msgs.msg.SolidPrimitive.BOX
        constraint_region.dimensions = [float('inf'), float('inf'), float('inf')]  # 只限制Z方向的最小尺寸

        # 设置约束框的位置，确保z坐标大于0.001
        box_pose = geometry_msgs.msg.Pose()
        box_pose.position.z = 0.001  # 设置约束区域的底部z坐标为0.001
        box_pose.orientation.w = 1.0

        # 将约束区域和位姿添加到位置约束
        position_constraint.constraint_region.primitives.append(constraint_region)
        position_constraint.constraint_region.primitive_poses.append(box_pose)
        position_constraint.weight = 1.0

        # 将位置约束添加到约束集
        constraints.position_constraints.append(position_constraint)
        constraints.name = "z_above_0.001"

        # 设置路径约束
        self.move_group.set_path_constraints(constraints)
        
        rospy.loginfo("Constraints added: z > 0.001")

    def verify_constraints(self):
        """验证当前设置的路径约束"""
        constraints = self.move_group.get_path_constraints()
        if constraints:
            rospy.loginfo("Current constraints:")
            rospy.loginfo(constraints)
            if constraints.position_constraints:
                rospy.loginfo("Position constraints exist")
                return True
        else:
            rospy.loginfo("No constraints set")
            return False
        
    def test_constraints(self):
        """测试约束是否生效"""
        # 添加约束
        self.add_constraints()
        
        # 尝试规划到一个z < 0.01的位置
        test_pose = geometry_msgs.msg.Pose()
        test_pose.position.x = 0.4
        test_pose.position.y = 0.0
        test_pose.position.z = -0.005  # 违反z>0.001的约束
        test_pose.orientation.w = 1.0
        
        self.move_group.set_pose_target(test_pose)
        success = self.move_group.plan()[0]
        
        if not success:
            rospy.loginfo("Constraints working - prevented planning to z < 0.001")
        else:
            rospy.logwarn("Constraints may not be working - was able to plan below z = 0.001")
        
        self.move_group.clear_pose_targets()
        return not success
    
    def clear_constraints(self):
        """清除路径约束"""
        self.move_group.clear_path_constraints()
        rospy.loginfo("Path constraints cleared.")

    def move(self, position, rpy):
        """根据目标位置和姿态移动机器人"""
        try:
            # 添加路径约束
            self.add_constraints()

            # 首先检查目标位置是否满足约束
            if position[2] < 0.001:  # 检查Z轴约束
                rospy.logerr(f"Target position z={position[2]} violates minimum height constraint")
                return False

            # 将 RPY 转换为四元数
            quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            # 创建目标位姿
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.position.x = position[0]
            pose_goal.position.y = position[1]
            pose_goal.position.z = position[2]
            pose_goal.orientation.x = quaternion[0]
            pose_goal.orientation.y = quaternion[1]
            pose_goal.orientation.z = quaternion[2]
            pose_goal.orientation.w = quaternion[3]

            # 设置规划参数
            self.move_group.set_planning_time(30.0)
            self.move_group.set_num_planning_attempts(30)
            self.move_group.set_max_velocity_scaling_factor(0.1)
            self.move_group.set_max_acceleration_scaling_factor(0.1)

            # 设置目标位置并进行路径规划
            self.move_group.set_pose_target(pose_goal)
            success = self.move_group.plan()  # 解包元组

            if not success:
                rospy.logerr("Motion planning failed. No valid plan generated.")
                return False

            # 执行规划路径
            success = self.move_group.go(wait=True)
            
            if not success:
                rospy.logerr("Move execution failed")
                return False

            rospy.loginfo(f"Move successful to position: {position} and RPY: {rpy}")
            return True

        except Exception as e:
            rospy.logerr(f"Error in move operation: {e}")
            return False
        finally:
            # 清除目标并停止运动
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.clear_constraints()

    def grasp_approach(self, start_position, end_position, rpy):
        """
        Approach the target position from the starting position while maintaining the end-effector's orientation.
        Use MoveIt's computeCartesianPath for Cartesian path planning.
        """
        try:
            # Convert RPY (Roll, Pitch, Yaw) to quaternion
            quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            # Set the start position and target orientation
            start_pose = geometry_msgs.msg.Pose()
            start_pose.position.x = start_position[0]
            start_pose.position.y = start_position[1]
            start_pose.position.z = start_position[2]
            start_pose.orientation.x = quaternion[0]
            start_pose.orientation.y = quaternion[1]
            start_pose.orientation.z = quaternion[2]
            start_pose.orientation.w = quaternion[3]

            end_pose = geometry_msgs.msg.Pose()
            end_pose.position.x = end_position[0]
            end_pose.position.y = end_position[1]
            end_pose.position.z = end_position[2]
            end_pose.orientation.x = quaternion[0]
            end_pose.orientation.y = quaternion[1]
            end_pose.orientation.z = quaternion[2]
            end_pose.orientation.w = quaternion[3]

            # Set the list of waypoints (start and end poses)
            waypoints = [start_pose, end_pose]

            self.move_group.set_planning_time(30.0)

            # Use computeCartesianPath to plan the Cartesian path
            rospy.loginfo("Planning Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,   # List of waypoints
                0.1,        # Maximum step size
                True,        # Enable collision checking
                0.0          # Jump threshold (0.0 means no jumping)
            )

            # Check the success of the path planning
            if fraction < 1.0:
                rospy.logwarn(f"Path planning succeeded for only {fraction * 100:.2f}% of the path")
                return False

            rospy.loginfo("Path planning completed successfully!")

            # Execute the planned path
            rospy.loginfo("Executing Cartesian path...")
            success = self.move_group.execute(plan, wait=True)

            # Stop and clean up
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            if success:
                rospy.loginfo("Grasp approach executed successfully.")
            else:
                rospy.logwarn("Grasp approach execution failed.")

            return success

        except Exception as e:
            rospy.logerr(f"Exception in grasp_approach method: {e}")
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
        # rospy.loginfo("Getting current pose...")
        # current_pose = robot_mover.get_current_pose()
        # if current_pose:
        #     rospy.loginfo("Current position and orientation retrieved successfully.")
        #     print(current_pose)

        # # 1. 验证约束设置
        # robot_mover.add_constraints()
        # if robot_mover.verify_constraints():
        #     rospy.loginfo("Constraints set successfully")
        
        # # 2. 测试约束效果
        # if robot_mover.test_constraints():
        #     rospy.loginfo("Constraints preventing invalid movements")

        # 初始和目标位置
        start_position = [0.4, 0, 0.5]
        end_position = [0.3, 0.0, 0.2]  # 修改为合理的z值
        target_rpy = [0, np.pi, np.pi]
        robot_mover.move(end_position, target_rpy)

        rospy.loginfo("Starting grasp approach...")
        # robot_mover.grasp_approach(start_position, end_position, target_rpy)
        
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")