#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler
import numpy as np
import moveit_msgs.msg
import shape_msgs.msg
import visualization_msgs.msg
import geometry_msgs.msg
import tf

class TargetFrameBroadcaster:
    def __init__(self, reference_frame="world", target_frame="target_frame"):
        """
        Initializes the TargetFrameBroadcaster.
        :param reference_frame: The reference frame (default: "world")
        :param target_frame: The target frame to broadcast (default: "target_frame")
        """
        self.reference_frame = reference_frame
        self.target_frame = target_frame
        
        # 创建tf广播器
        self.tf_broadcaster = tf.TransformBroadcaster()

        # 初始化目标位置和姿态
        self.last_position = [0.0, 0.0, 0.0]  # 初始位置
        self.last_quaternion = [0.0, 0.0, 0.0, 1.0]  # 初始姿态（单位四元数）
        self.target_position = self.last_position
        self.target_quaternion = self.last_quaternion

        # 设置定时器，每秒发布一次目标位置
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_target_transform)

    def publish_target_transform(self, event):
        """每秒发布目标位置和姿态"""
        try:
            # 如果目标位置发生变化，则更新
            if not np.array_equal(self.target_position, self.last_position) or not np.array_equal(self.target_quaternion, self.last_quaternion):
                # 发布目标坐标变换
                self.tf_broadcaster.sendTransform(
                    (self.target_position[0], self.target_position[1], self.target_position[2]),
                    self.target_quaternion,
                    rospy.Time.now(),
                    self.target_frame,  # 目标坐标系
                    self.reference_frame  # 参考坐标系（例如 "world"）
                )
                # 更新之前的目标位置和姿态
                self.last_position = self.target_position
                self.last_quaternion = self.target_quaternion
            else:
                # 如果没有更新，继续发布之前的数据
                self.tf_broadcaster.sendTransform(
                    (self.last_position[0], self.last_position[1], self.last_position[2]),
                    self.last_quaternion,
                    rospy.Time.now(),
                    self.target_frame,
                    self.reference_frame
                )
        except Exception as e:
            rospy.logerr(f"Failed to publish target transform: {e}")

    def update_target(self, position, rpy):
        """更新目标位置和姿态"""
        try:
            # 将RPY角度转换为四元数
            quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            # 更新目标位置和姿态
            self.target_position = position
            self.target_quaternion = quaternion

            rospy.loginfo(f"Target position set to: {position} and RPY: {rpy}")
            return True
        except Exception as e:
            rospy.logerr(f"Error in update_target operation: {e}")
            return False

    def stop(self):
        """停止定时器"""
        self.timer.shutdown()


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
            self.add_wall(wall_name="wall_right", wall_position=[0.0, 0.8, 0.0], theta=-np.pi / 4)
            self.add_wall(wall_name="wall_left", wall_position=[0.0, -0.8, 0.0], theta=np.pi / 4)
            rospy.loginfo("MoveRobot initialized successfully.")
            self.init_joint_values = self.get_current_joint_values()
        except Exception as e:
            rospy.logerr(f"Error initializing MoveRobot: {e}")
            raise
        self.gazebo_init_joints_c = [0.00020990855484193105, -0.7795008908903167, 0.00010624718470086947, -2.3667376031479828, -0.0002316322324373843, 1.573599783202095, 0.7853073403622508]
        
        # 初始化TargetFrameBroadcaster
        self.target_broadcaster = TargetFrameBroadcaster(reference_frame="world", target_frame="target_frame")


    def add_wall(self, wall_name="wall", wall_position=[2.0, 0.0, 1.0], theta=np.pi / 5):
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
            rospy.loginfo(
                f"Wall '{wall_name}' added to the scene with rotation theta={theta} radians at position: {wall_pose.position.x}, {wall_pose.position.y}, {wall_pose.position.z}")
        except Exception as e:
            rospy.logerr(f"Error adding wall: {e}")

    def add_table(self,table_name="fixed_table", z=0.001):
        """Add a table (box) object to prevent the robot from moving below a certain height"""
        try:
            # Define the table's size and position
            table = moveit_commander.PlanningSceneInterface()
            table_size = [4.0, 4.0, 0.001]  # Table is 2m x 1m with 0.001m height
            table_pose = geometry_msgs.msg.Pose()
            table_pose.position.x = 0.0
            table_pose.position.y = 0.0
            table_pose.position.z = z

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

    def remove_table(self, table_name="constraint_table"):
        """Remove the table (or any object) from the planning scene"""
        try:
            planning_scene = moveit_commander.PlanningSceneInterface()
            planning_scene.remove_world_object(table_name)  # Remove the table by its name
            rospy.loginfo(f"Table '{table_name}' removed from the planning scene.")
        except Exception as e:
            rospy.logerr(f"Error removing table: {e}")

    def move(self, position, rpy, add_privant_table=True, retry_init=False):
        # retry用于当move失败时，恢复初始状态，再试一次
        """Move the robot based on the target position and orientation"""
        rnt = True
        try:
            # 防止撞倒其他cube！
            if add_privant_table:
                self.add_table("constraint_table", 0.06)
            # First check if the target position satisfies the constraints
            if position[2] < 0.001:  # Check Z-axis constraint
                rospy.logerr(f"Target position z={position[2]} violates minimum height constraint")
                return False
            self.target_broadcaster.update_target(position, rpy)
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
            self.move_group.set_max_velocity_scaling_factor(1)
            self.move_group.set_max_acceleration_scaling_factor(1)

            # Set target position and plan the path
            self.move_group.set_pose_target(pose_goal)
            # Change: Fix plan return value handling to ensure valid plans and avoid wall collisions
            plan, _, _, error_code = self.move_group.plan()
            if not plan or error_code.val != 1:  # Success code is 1
                rospy.logerr("Motion planning failed. No valid plan generated.")
                rnt = False

            # Execute the planned path
            success = self.move_group.go(wait=True)

            if not success:
                rospy.logerr("Move execution failed")
                rnt = False


        except Exception as e:
            rospy.logerr(f"Error in move operation: {e}")
            rnt =  False
        finally:
            # Clear targets and stop movement
            self.remove_table("constraint_table")
            self.move_group.stop()
            self.move_group.clear_pose_targets()

        self.remove_table("constraint_table")
        if rnt == True:
            rospy.loginfo(f"Move successful to position: {position} and RPY: {rpy}")
        else:
            if retry_init:
                self.restore_init_joint_c_gazebo()
                print("move 失败，垂死挣扎一次！")
                rnt = self.move(position, rpy, add_privant_table, retry_init=False) # 只能尝试一次！
                if rnt == False:
                    print("你移动了个什么玩意，初始状态也到不了！")
                    print("move的position, rpy 为", position, rpy, " 自己debug去吧！")
                else:
                    print("move垂死挣扎成功了，niubi!")
            
        return rnt
        


    def grasp_approach(self, start_position, end_position, rpy, z_min=0.001, max_retries=10, retry_init=False):
        """
        Approach the target position from the starting position while maintaining the end-effector's orientation.
        Use MoveIt's computeCartesianPath for Cartesian path planning.
        If path planning fails, retry up to max_retries times.
        """
        rnt = True
        try:
            # Change: Add z_min checks for start_position and end_position to prevent low movements near walls
            if start_position[2] < z_min or end_position[2] < z_min:
                rospy.logerr("Start or end position violates z_min constraint")
                rnt = False

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
            self.target_broadcaster.update_target(start_position, rpy)

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
                # Change: Reduce step size for better collision checking with walls
                (plan, fraction) = self.move_group.compute_cartesian_path(
                    waypoints,  # List of waypoints
                    0.01,  # Reduced from 0.1 to 0.02 for finer collision detection
                    False  # Enable collision checking
                )

                # Check the success of the path planning
                if fraction >= 1.0:
                    rospy.loginfo("Path planning completed successfully!")
                    break
                else:
                    rospy.logwarn(f"Path planning succeeded for only {fraction * 100:.2f}% of the path")
                    if attempt == max_retries - 1:
                        rospy.logerr("Maximum retries reached. Path planning failed.")
                        rnt = False
                    rospy.loginfo("Retrying path planning...")
            self.target_broadcaster.update_target(end_position, rpy)


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
                    rnt = False

            else:
                rospy.logerr("Path planning failed after multiple retries.")
                rnt = False
            rnt = success

        except Exception as e:
            rospy.logerr(f"Exception in grasp_approach method: {e}")
            rnt = False
        
        if retry_init and rnt == False:
            self.restore_init_joint_c_gazebo()
            print("approach 失败，垂死挣扎一次！")
            rnt = self.grasp_approach(start_position, end_position, rpy, z_min, max_retries, retry_init=False) # 只能尝试一次！
            if rnt == False:
                print("你移动了个什么玩意，初始状态也到不了！")
                print("approach的start_position, end_position, rpy 为", start_position, end_position, rpy, " 自己debug去吧！")
            else:
                print("垂死挣扎成功了，niubi!")

        return rnt

    def get_current_pose(self):
        """Get the current position and orientation of the end effector"""
        try:
            current_pose = self.move_group.get_current_pose().pose
            position = current_pose.position
            orientation = current_pose.orientation
            rospy.loginfo(
                "Current pose: Position({:.3f}, {:.3f}, {:.3f}), Orientation({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(
                    position.x, position.y, position.z,
                    orientation.x, orientation.y, orientation.z, orientation.w
                ))
            return current_pose
        except Exception as e:
            rospy.logerr(f"Failed to get current pose: {e}")
            return None

    def __del__(self):
        if self.target_broadcaster:
            self.target_broadcaster.stop()
        # self.restore_initial_joint_values()
        self.restore_init_joint_c_gazebo()
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

    def restore_init_joint_c_gazebo(self):
        # 恢复从gazebo的初始位置
        self.move_group.set_joint_value_target(self.gazebo_init_joints_c)
        success = self.move_group.go(wait=True)


if __name__ == "__main__":
    try:
        robot_mover = MoveRobot()
        # robot_mover.restore_init_joint_c_gazebo()

        # Initial and target positions
        start_position = [0.40, 0.11, 0.02]
        end_position = [0.40, 0.11, 0.6]  # Modified to a valid z value
        target_rpy = [0, np.pi, np.pi]
        # robot_mover.move(start_position, target_rpy, add_privant_table=False)

        position = [0.6, -0.14, 0.23] 
        rpy = [0, 3.141592653589793, 0.7853981633974483]
        robot_mover.move(start_position, target_rpy, add_privant_table=False, retry_init=True)
        # rospy.loginfo("Starting grasp approach...")
        for i in range(10): 
            robot_mover.grasp_approach(start_position, end_position, target_rpy)
            robot_mover.grasp_approach(end_position, start_position, target_rpy)


        # while True:
        #     print("debuging! terminate the process youself per hand!")
        #     pass

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
