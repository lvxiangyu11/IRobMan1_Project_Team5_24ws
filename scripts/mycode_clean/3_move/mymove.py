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
            # self.add_constraints()
            rospy.loginfo("MoveRobot initialized successfully.")
        except Exception as e:
            rospy.logerr(f"Error initializing MoveRobot: {e}")
            raise

    def add_constraints(self, z_min=0.001):
        """Add path constraints to restrict the robot's movement to the region where z > z_min"""
        constraints = moveit_commander.Constraints()

        # Create position constraint
        position_constraint = moveit_msgs.msg.PositionConstraint()
        position_constraint.header.frame_id = self.move_group.get_planning_frame()
        position_constraint.link_name = self.move_group.get_end_effector_link()

        # Define the region for the position constraint
        constraint_region = shape_msgs.msg.SolidPrimitive()
        constraint_region.type = shape_msgs.msg.SolidPrimitive.BOX
        constraint_region.dimensions = [float('inf'), float('inf'), float('inf')]  # Only restrict the Z direction's minimum size

        # Set the constraint box's position, ensuring that z-coordinate is greater than 0.001
        box_pose = geometry_msgs.msg.Pose()
        box_pose.position.z = z_min  # Set the bottom z-coordinate of the constraint region to 0.001
        box_pose.orientation.w = 1.0

        # Add the constraint region and pose to the position constraint
        position_constraint.constraint_region.primitives.append(constraint_region)
        position_constraint.constraint_region.primitive_poses.append(box_pose)
        position_constraint.weight = 1.0

        # Add position constraint to the constraints set
        constraints.position_constraints.append(position_constraint)
        constraints.name = "z_above_0.001" + str(z_min)

        # Set path constraints
        self.move_group.set_path_constraints(constraints)
        
        rospy.loginfo("Constraints added: z > " + str(z_min))

    def verify_constraints(self):
        """Verify the currently set path constraints"""
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
        """Test whether constraints are working"""
        # Add constraints
        self.add_constraints()
        
        # Try planning to a position where z < 0.01
        test_pose = geometry_msgs.msg.Pose()
        test_pose.position.x = 0.4
        test_pose.position.y = 0.0
        test_pose.position.z = -0.005  # Violating the z > 0.001 constraint
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
        """Clear path constraints"""
        self.move_group.clear_path_constraints()
        rospy.loginfo("Path constraints cleared.")

    def move(self, position, rpy, z_min=0.001):
        """Move the robot based on the target position and orientation"""
        try:
            # Add path constraints
            # self.add_constraints(z_min)

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
            self.move_group.set_planning_time(5.0)
            self.move_group.set_num_planning_attempts(30)
            self.move_group.set_max_velocity_scaling_factor(0.1)
            self.move_group.set_max_acceleration_scaling_factor(0.1)

            # Set target position and plan the path
            self.move_group.set_pose_target(pose_goal)
            success = self.move_group.plan()  # Unpack tuple
            
            self.clear_constraints()

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
            self.clear_constraints()
            return False
        finally:
            self.clear_constraints()
            # Clear targets and stop movement
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.clear_constraints()

    def grasp_approach(self, start_position, end_position, rpy, z_min=0.001, max_retries=10):
        """
        Approach the target position from the starting position while maintaining the end-effector's orientation.
        Use MoveIt's computeCartesianPath for Cartesian path planning.
        If path planning fails, retry up to max_retries times.
        """
        try:
            self.add_constraints(z_min)
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
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveRobot shut down.")


if __name__ == "__main__":
    try:
        robot_mover = MoveRobot()

        # Get current end effector pose
        # rospy.loginfo("Getting current pose...")
        # current_pose = robot_mover.get_current_pose()
        # if current_pose:
        #     rospy.loginfo("Current position and orientation retrieved successfully.")
        #     print(current_pose)

        # # 1. Verify constraints
        robot_mover.add_constraints()
        if robot_mover.verify_constraints():
            rospy.loginfo("Constraints set successfully")
        
        # # 2. Test constraint effectiveness
        if robot_mover.test_constraints():
            rospy.loginfo("Constraints preventing invalid movements")

        # Initial and target positions
        start_position = [0.4, 0, 0.5]
        end_position = [0.3, 0.0, 0.2]  # Modified to a valid z value
        target_rpy = [0, np.pi, np.pi]
        # robot_mover.move(end_position, target_rpy)

        rospy.loginfo("Starting grasp approach...")
        robot_mover.grasp_approach(start_position, end_position, target_rpy)
        
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
