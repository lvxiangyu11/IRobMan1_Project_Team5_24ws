#!/usr/bin/env python

import rospy
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal


class MyGripper:
    def __init__(self, restart=False):
        """Initialize the Gripper controller"""
        # Check if ROS node has been initialized
        if not rospy.core.is_initialized():
            rospy.init_node("my_gripper_node", anonymous=True)
        self.grasp_client = actionlib.SimpleActionClient(
            "/franka_gripper/grasp", GraspAction
        )
        self.move_client = actionlib.SimpleActionClient(
            "/franka_gripper/move", MoveAction
        )

        # Wait for Gripper action services to be available
        rospy.loginfo("Waiting for gripper action servers...")
        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()
        rospy.loginfo("Gripper action servers ready.")

    def close(self, width=0.04, inner=0.02, outer=0.02, speed=0.1, force=1.0):
        """
        Close the gripper to grasp an object
        :param width: Grasp width (m)
        :param inner: Inner tolerance range (m)
        :param outer: Outer tolerance range (m)
        :param speed: Grasp speed (m/s)
        :param force: Grasp force (N)
        """
        goal = GraspGoal()
        goal.width = width
        goal.epsilon.inner = inner
        goal.epsilon.outer = outer
        goal.speed = speed
        goal.force = force

        rospy.loginfo(f"Sending grasp goal: {goal}")
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result()
        result = self.grasp_client.get_result()

        if result.success:
            rospy.loginfo("Grasp successful.")
        else:
            rospy.logwarn("Grasp failed.")
        return result.success

    def open(self, width=0.08, speed=0.1):
        """
        Open the gripper to release an object
        :param width: Open width (m)
        :param speed: Open speed (m/s)
        """
        goal = MoveGoal()
        goal.width = width
        goal.speed = speed

        rospy.loginfo(f"Sending open goal: {goal}")
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result()
        result = self.move_client.get_result()

        if result.success:
            rospy.loginfo("Gripper opened successfully.")
        else:
            rospy.logwarn("Failed to open gripper.")
        return result.success


if __name__ == "__main__":
    try:
        gripper = MyGripper()

        rospy.loginfo("Closing gripper...")
        gripper.close(width=0.04, inner=0.02, outer=0.02, speed=0.1, force=1.0)

        rospy.loginfo("Opening gripper...")
        gripper.open(width=0.08, speed=0.1)

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")
