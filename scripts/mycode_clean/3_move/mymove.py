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
        # Initialize MoveIt and ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        if not rospy.core.is_initialized():
            rospy.init_node("my_gripper_node", anonymous=True)

        # Wait for move_group action server
        rospy.loginfo("Waiting for move_group action server...")
        try:
            # Initialize robot interface
            self.robot = moveit_commander.RobotCommander()
            self.group_name = "panda_manipulator"  # Adjust according to your robot
            self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
            self.add_table()
            self.add_wall(wall_name="wall_right", wall_position=[0.0, 0.8, 0.0], theta=-np.pi/5)
            self.add_wall(wall_name="wall_left", wall_position=[0.0, -0.8, 0.0], theta=np.pi/5)
            rospy.loginfo("MoveRobot initialized successfully.")
            self.init_joint_values = self.get_current_joint_values()
        except Exception as e:
            rospy.logerr(f"Error initializing MoveRobot: {e}")
            raise

    def add_wall(self, wall_name="wall", wall_position=[2.0, 0.0, 1.0], theta=np.pi/5):
        """Add a wall (box) object with rotation, position, and custom name"""
        try:
            # Define the wall's size
            wall_size = [0.1, 3.0, 4.0]  # Wall is 0.1m thick, 5m long, 2m high
            wall_pose = geometry_msgs.msg.Pose()

            # Set wall's position
            wall_pose.position.x = wall_position[0]  # x position from parameter
            wall_pose.position.y = wall_position[1]  # y position from parameter
            wall_pose.position.z = wall_position[2]  # z position from parameter

            # Convert rotation angle theta (in radians) to quaternion
            quaternion = quaternion_from_euler(0, 0, theta)  # Rotate around z-axis by theta
            wall_pose.orientation.x = quaternion[0]
            wall_pose.orientation.y = quaternion[1]
            wall_pose.orientation.z = quaternion[2]
            wall_pose.orientation.w = quaternion[3]

            # Create a box primitive to represent the wall
            wall_box = shape_msgs.msg.SolidPrimitive()
            wall_box.type = shape_msgs.msg.SolidPrimitive.BOX
            wall_box.dimensions = wall_size

            # Create a collision object for the wall
            wall_object = moveit_commander.CollisionObject()
            wall_object.header.frame_id = self.move_group.get_planning_frame()
            wall_object.id = wall_name  # Set the custom name from the parameter
            wall_object.primitives.append(wall_box)
            wall_object.primitive_poses.append(wall_pose)

            # Add the collision object (wall) to the planning scene
            table = moveit_commander.PlanningSceneInterface()
            table.add_object(wall_object)
            rospy.loginfo(f"Wall '{wall_name}' added to the scene with rotation theta={theta} radians at position: {wall_pose.position.x}, {wall_pose.position.y}, {wall_pose.position.z}")
        except Exception as e:
            rospy.logerr(f"Error adding wall: {e}")

    def add_table(self):
        """Add a table (box) object to prevent the robot from moving below a certain height"""
        try:
            # Define the table's size and position
            table = moveit_commander.PlanningSceneInterface()
            table_name = "table"
            table_size = [4.0, 4.0, 0.001]  # Table is 2m x 1m with 0.001m height
            table_pose = geometry_msgs.msg.Pose()
            table_pose.position.x = 0.0
            table_pose.position.y = 0.0
            table_pose.position.z = 0.001  # The top of the table is at z = 0.001m

            # Create a box primitive to represent the table
            table_box = shape_msgs.msg.SolidPrimitive()
            table_box.type = shape_msgs.msg.SolidPrimitive.BOX
            table_box.dimensions = table_size

            # Create a collision object for the table
            table_object = moveit_commander.CollisionObject()
            table_object.header.frame_id = self.move_group.get_planning_frame()
            table_object.id = table_name
            table_object.primitives.append(table_box)
            table_object.primitive_poses.append(table_pose)

            # Apply the collision object to the planning scene
            table.add_object(table_object)
            rospy.loginfo(f"Table added to the scene to prevent collision below z = 0.001")
        except Exception as e:
            rospy.logerr(f"Error adding table: {e}")


    def move(self, position, rpy, z_min=0.001):
        """Move the robot based on the target position and orientation"""
        try:
            # First check if the target position satisfies the constraints
            if position[2] < 0.001:  # Check Z-axis constraint
                rospy.logerr(f"Target position z={position[2]} violates minimum height constraint")
                return False

            # Convert RPY to quaternion
            quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            # Create target pose
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.position.x = position[0]
            pose_goal.position.y = position[1]
            pose_goal.position.z = position[2]
            pose_goal.orientation.x = quaternion[0]
            pose_goal.orientation.y = quaternion[1]
            pose_goal.orientation.z = quaternion[2]
            pose_goal.orientation.w = quaternion[3]

            # Set planning parameters
            self.move_group.set_planning_time(25.0)
            self.move_group.set_num_planning_attempts(30)
            self.move_group.set_max_velocity_scaling_factor(0.1)
            self.move_group.set_max_acceleration_scaling_factor(0.1)

            # Set target position and plan the path
            self.move_group.set_pose_target(pose_goal)
            success = self.move_group.plan()  # Unpack tuple
            
            if not success:
                rospy.logerr("Motion planning failed. No valid plan generated.")
                return False

            # Execute the planned path
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
            # Clear targets and stop movement
            self.move_group.stop()
            self.move_group.clear_pose_targets()

    def grasp_approach(self, start_position, end_position, rpy, z_min=0.001, max_retries=10):
        """
        Approach the target position from the starting position while maintaining the end-effector's orientation.
        Use MoveIt's computeCartesianPath for Cartesian path planning.
        If path planning fails, retry up to max_retries times.
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

            self.move_group.set_planning_time(5.0)

            # Retry logic for planning
            for attempt in range(max_retries):
                rospy.loginfo(f"Attempt {attempt + 1} to plan Cartesian path...")
                # Use computeCartesianPath to plan the Cartesian path
                (plan, fraction) = self.move_group.compute_cartesian_path(
                    waypoints,   # List of waypoints
                    0.1,         # Maximum step size
                    True,        # Enable collision checking
                    0.0          # Jump threshold (0.0 means no jumping)
                )

                # Check the success of the path planning
                if fraction >= 1.0:
                    rospy.loginfo("Path planning completed successfully!")
                    break
                else:
                    rospy.logwarn(f"Path planning succeeded for only {fraction * 100:.2f}% of the path")
                    if attempt == max_retries - 1:
                        rospy.logerr("Maximum retries reached. Path planning failed.")
                        return False
                    rospy.loginfo("Retrying path planning...")

            # Execute the planned path if successful
            if fraction >= 1.0:
                rospy.loginfo("Executing Cartesian path...")
                success = self.move_group.execute(plan, wait=True)

                # Stop and clean up
                self.move_group.stop()
                self.move_group.clear_pose_targets()

                if success:
                    rospy.loginfo("Grasp approach executed successfully.")
                else:
                    rospy.logwarn("Grasp approach execution failed.")
                    return False
            else:
                rospy.logerr("Path planning failed after multiple retries.")
                return False

            return success

        except Exception as e:
            rospy.logerr(f"Exception in grasp_approach method: {e}")
            return False

    def get_current_pose(self):
        """Get the current position and orientation of the end effector"""
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
        self.restore_initial_joint_values()
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveRobot shut down.")

    def get_current_joint_values(self):
        """获取机器人的当前关节配置"""
        try:
            joint_values = self.move_group.get_current_joint_values()
            rospy.loginfo(f"Current joint values: {joint_values}")
            return joint_values
        except Exception as e:
            rospy.logerr(f"Failed to get current joint values: {e}")
            return None

    def restore_initial_joint_values(self):
        """恢复到初始关节配置"""
        try:
            if self.init_joint_values is not None:
                rospy.loginfo("Restoring to initial joint values...")
                # 设置目标关节配置
                self.move_group.set_joint_value_target(self.init_joint_values)
                
                # 规划并执行动作
                success = self.move_group.go(wait=True)
                if success:
                    rospy.loginfo("Successfully restored to the initial joint configuration.")
                else:
                    rospy.logerr("Failed to restore to initial joint configuration.")
            else:
                rospy.logerr("Initial joint values are not defined.")
        except Exception as e:
            rospy.logerr(f"Error restoring initial joint values: {e}")

if __name__ == "__main__":
    try:
        robot_mover = MoveRobot()

        # Get current end effector pose
        # rospy.loginfo("Getting current pose...")
        # current_pose = robot_mover.get_current_pose()
        # if current_pose:
        #     rospy.loginfo("Current position and orientation retrieved successfully.")
        #     print(current_pose)

        # Initial and target positions
        start_position = [0.4, 0, 0.5]
        end_position = [0.7, 0.0, 0.02]  # Modified to a valid z value
        target_rpy = [0, np.pi, np.pi]
        # target_rpy = [-3.1386386530494828, 0.003954958007977849, -3.0987328681412816]
        # end_position = [0.7, -0.5, 0.02]
        robot_mover.move(end_position, target_rpy)

        rospy.loginfo("Starting grasp approach...")
        # robot_mover.grasp_approach(start_position, end_position, target_rpy)
        
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
