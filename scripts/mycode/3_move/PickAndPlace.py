#!/usr/bin/env python

import numpy as np
import rospy
from mygripper import MyGripper
from mymove import MoveRobot

class PickAndPlace:
    def __init__(self, approach_distance=0.1):
        self.robot_mover = MoveRobot()
        self.gripper = MyGripper()
        self.approach_distance = approach_distance

    def pick_and_place(self, 
                        pick_pos, pick_rpy,  # 物体当前位置
                        place_pos, place_rpy):  # 物体目标位置
        try:
            # 打开夹爪
            self.gripper.open(width=0.08, speed=0.1)

            # 移动到物体上方高位置
            high_pick_pos = self._calculate_approach_position(pick_pos)
            self.robot_mover.move(high_pick_pos, pick_rpy)

            # 移动到物体位置
            self.robot_mover.grasp_approach(high_pick_pos, pick_pos, pick_rpy)

            # 闭合夹爪
            self.gripper.close(width=0.05, inner=0.01, outer=0.01, speed=0.1, force=50.0)

            # 移动到高位置
            high_pick_up_pos = self._calculate_approach_position(pick_pos)
            self.robot_mover.grasp_approach(pick_pos, high_pick_up_pos, pick_rpy)

            # 移动到放置高位置
            high_place_pos = self._calculate_approach_position(place_pos)
            self.robot_mover.grasp_approach(high_pick_up_pos, high_place_pos, place_rpy)

            # 移动到放置位置
            self.robot_mover.grasp_approach(high_place_pos, place_pos, place_rpy)

            # 打开夹爪释放
            self.gripper.open(width=0.08, speed=0.1)

            # 返回高位置
            self.robot_mover.grasp_approach(place_pos, high_place_pos, place_rpy)

        except Exception as e:
            rospy.logerr(f"Error in pick and place: {e}")

    def _calculate_approach_position(self, pos):
        approach_pos = list(pos)
        approach_pos[2] += self.approach_distance
        return approach_pos

def test():
    rospy.init_node('pick_and_place_node', anonymous=True)

    pick_place = PickAndPlace(approach_distance=0.1)
    pick_pos = [0.4, 0, 0.13]      # 物体初始位置
    pick_rpy = [0, np.pi, np.pi/2 + np.pi/4]
        
    place_pos = [0.5, 0.1, 0.13]   # 物体目标位置
    place_rpy = [0, np.pi, np.pi/2 + np.pi/4]
    try:
        for i in range(100):
            pick_place.pick_and_place(
                pick_pos=pick_pos,
                pick_rpy=pick_rpy,
                place_pos=place_pos,
                place_rpy=place_rpy
            )
            pick_pos, place_pos = place_pos, pick_pos

    except rospy.ROSInterruptException:
        pass

def peral():
    # TODO: Grasp Generation, Collision Avoidance by Moveit
    rospy.init_node('pick_and_place_node', anonymous=True)
    pick_place = PickAndPlace(approach_distance=0.1)
    pick_rpy = [0, np.pi, np.pi/4]
    place_rpy = [0, np.pi, np.pi/4]

    # P
    pick_place.pick_and_place(
        pick_pos=[0.6, -0.14, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, 0.1, 0.13],
        place_rpy=place_rpy
    )

    # E
    pick_place.pick_and_place(
        pick_pos=[0.5, 0.22, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, 0.05, 0.13],
        place_rpy=place_rpy
    )

    # R
    pick_place.pick_and_place(
        pick_pos=[0.5, -0.02, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, 0.00, 0.13],
        place_rpy=place_rpy
    )

    # A
    pick_place.pick_and_place(
        pick_pos=[0.6, -0.08, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, -0.05, 0.13],
        place_rpy=place_rpy
    )

    # L
    pick_place.pick_and_place(
        pick_pos=[0.6, 0.28, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, -0.1, 0.13],
        place_rpy=place_rpy
    )
    pass

if __name__ == "__main__":
    peral()