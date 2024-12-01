#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def control_gripper(open_gripper=True):
    rospy.init_node('gripper_control_node')
    pub = rospy.Publisher('/effort_joint_trajectory_controller/command', JointTrajectory, queue_size=10)

    trajectory = JointTrajectory()
    trajectory.joint_names = ['panda_finger_joint1', 'panda_finger_joint2']

    point = JointTrajectoryPoint()
    if open_gripper:
        point.positions = [0.04, 0.04]  # 抓手张开
    else:
        point.positions = [0.0, 0.0]    # 抓手闭合
    point.effort = [10.0, 10.0]  # 力矩
    point.time_from_start = rospy.Duration(1.0)  # 持续时间

    trajectory.points.append(point)
    pub.publish(trajectory)
    rospy.loginfo("Gripper command sent!")

if __name__ == '__main__':
    try:
        control_gripper(open_gripper=True)  # True: 张开抓手, False: 闭合抓手
    except rospy.ROSInterruptException:
        pass
