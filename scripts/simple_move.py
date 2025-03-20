#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler
import numpy as np
import moveit_msgs.msg
import shape_msgs.msg

def test_arm_movement():
    # Initialize moveit_commander and a rospy node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_arm_test', anonymous=True)
    
    # Instantiate a RobotCommander object
    robot = moveit_commander.RobotCommander()
    
    # Instantiate a PlanningSceneInterface object
    scene = moveit_commander.PlanningSceneInterface()
    
    # Instantiate a MoveGroupCommander object for the panda_arm group
    arm_group = moveit_commander.MoveGroupCommander("panda_arm")
    
    # Print current robot state
    print("============ Robot state: ")
    print(robot.get_current_state())
    
    # Get current joint values
    current_joints = arm_group.get_current_joint_values()
    print("Current joint values:", current_joints)
    
    # Set a target position
    target_joints = list(current_joints)  # Start with current position
    target_joints[0] = 0.5  # Move the first joint to 0.5 radians
    
    # Set the joint target
    arm_group.set_joint_value_target(target_joints)
    
    # Plan and execute the trajectory
    print("Planning and executing trajectory...")
    
    # Check your MoveIt version and use the appropriate method
    try:
        # For older versions of MoveIt
        plan = arm_group.plan()
        
        # In older versions, plan() returns the trajectory directly
        if plan and len(plan.joint_trajectory.points) > 0:
            print("Plan found, executing...")
            arm_group.execute(plan)
        else:
            print("No plan found")
            
    except AttributeError:
        try:
            # For newer versions of MoveIt
            plan_success, plan, planning_time, error_code = arm_group.plan()
            
            if plan_success:
                print("Plan found, executing...")
                arm_group.execute(plan)
            else:
                print("No plan found. Error code:", error_code)
                
        except ValueError:
            # Another method to try
            print("Using alternative planning method...")
            plan = arm_group.plan()
            if isinstance(plan, tuple):
                # Unpacking the tuple
                success = plan[0]
                trajectory = plan[1]
                if success:
                    print("Plan found, executing...")
                    arm_group.execute(trajectory)
                else:
                    print("No plan found")
            else:
                # Just try to execute what we got
                arm_group.execute(plan)
    
    # Print new joint values after movement
    rospy.sleep(2)  # Give time for execution
    new_joints = arm_group.get_current_joint_values()
    print("New joint values:", new_joints)
    
    # Clean up
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        test_arm_movement()
        print("Movement test completed")
    except rospy.ROSInterruptException:
        print("Movement test interrupted")
    except Exception as e:
        print(f"Error: {e}")