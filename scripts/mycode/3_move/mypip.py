import numpy as np
from mygripper import MyGripper
from mymove import MoveRobot
import rospy

# 初始化机器人和夹爪
robot_mover = MoveRobot()
gripper = MyGripper()

# 定义目标姿态
target_rpy = [0, np.pi, np.pi / 2 + np.pi / 4]  # 目标姿态 (roll, pitch, yaw)

for i in range(100):
    rospy.loginfo(f"Iteration {i + 1}/100: Starting cycle.")

    # 移动到上方初始位置
    start_position = [0.4, 0, 0.13 + 0.10]  # 高位置
    robot_mover.move(start_position, target_rpy)

    # 打开夹爪
    gripper.open(width=0.08, speed=0.1)

    # 使用 grasp_approach 移动到抓取位置
    start_position = [0.4, 0, 0.13 + 0.10]  # 高位置
    target_position = [0.4, 0, 0.13 + 0.00]  # 抓取位置
    robot_mover.grasp_approach(start_position, target_position, target_rpy)

    # 闭合夹爪抓取物体
    gripper.close(width=0.05, inner=0.01, outer=0.01, speed=0.1, force=50.0)

    # 使用 grasp_approach 移动到上方释放位置
    start_position = [0.4, 0, 0.13 + 0.00]  # 抓取位置
    target_position = [0.4, 0, 0.13 + 0.10]  # 高位置
    robot_mover.grasp_approach(start_position, target_position, target_rpy)

    # 使用 grasp_approach 再次移动到释放位置
    start_position = [0.4, 0, 0.13 + 0.10]  # 高位置
    target_position = [0.4, 0, 0.13 + 0.00]  # 释放位置
    robot_mover.grasp_approach(start_position, target_position, target_rpy)

    # 打开夹爪释放物体
    gripper.open(width=0.08, speed=0.1)

    # 使用 grasp_approach 移动回到上方初始位置
    start_position = [0.4, 0, 0.13 + 0.00]  # 释放位置
    target_position = [0.4, 0, 0.13 + 0.10]  # 高位置
    robot_mover.grasp_approach(start_position, target_position, target_rpy)

    rospy.loginfo(f"Iteration {i + 1}/100: Cycle completed.")

rospy.loginfo("All 100 iterations completed.")
