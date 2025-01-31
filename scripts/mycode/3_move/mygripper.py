#!/usr/bin/env python

import rospy
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal


class MyGripper:
    def __init__(self):
        """初始化 Gripper 控制器"""
        # 判断是否已经初始化 ROS 节点
        if not rospy.core.is_initialized():
            rospy.init_node("my_gripper_node", anonymous=True)
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)

        # 等待 Gripper 动作服务可用
        rospy.loginfo("Waiting for gripper action servers...")
        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()
        rospy.loginfo("Gripper action servers ready.")

    def close(self, width=0.05, inner=0.01, outer=0.01, speed=0.05, force=100.0):
        """
        闭合机械手抓取物体
        :param width: 抓取宽度 (m)
        :param inner: 内部容忍范围 (m)
        :param outer: 外部容忍范围 (m)
        :param speed: 抓取速度 (m/s)
        :param force: 抓取力 (N)
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
        打开机械手释放物体
        :param width: 打开宽度 (m)
        :param speed: 打开速度 (m/s)
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
        # gripper.close(width=0.05, inner=0.01, outer=0.01, speed=0.1, force=5.0)

        rospy.loginfo("Opening gripper...")
        gripper.open(width=0.08, speed=0.1)

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")
