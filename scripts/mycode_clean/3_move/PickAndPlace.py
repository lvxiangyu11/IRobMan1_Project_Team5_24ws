#!/usr/bin/env python

import numpy as np
import rospy
from mygripper import MyGripper
from mymove import MoveRobot
import time
import numpy as np


SAFE_POSITION = [0.50, -0.3, 0.02]
SAFE_ORIENTATION = [0, np.pi, np.pi]
RETRAY_STEPS = 2

class PickAndPlace:
    def __init__(self, approach_distance=0.20, restart=False):
        self.robot_mover = MoveRobot()
        self.gripper = MyGripper()
        self.approach_distance = approach_distance
        if restart:
            self.robot_mover.restore_init_joint_c_gazebo()

    def pick_and_place(
        self, pick_pos, pick_rpy, place_pos, place_rpy,  # Current position of the object
        extra_pick_patch=False # 是否尝试两个方向稳定抓取
    ):  # Target position of the object
        try:
            # Open the gripper
            self.gripper.open(width=0.08, speed=0.1)
            # time.sleep(0.2)

            # Move to a higher position above the object
            high_pick_pos = self._calculate_approach_position(pick_pos)
            self.robot_mover.move(high_pick_pos, pick_rpy, retry_init=True)
            # time.sleep(0.1)

            # Move to the object's position
            self.robot_mover.grasp_approach(high_pick_pos, pick_pos, pick_rpy, retry_init=True)
            # self.robot_mover.move(pick_pos, pick_rpy, retry_init=True)
            # time.sleep(0.2)

            # Close the gripper
            self.gripper.close(width=0.04, inner=0.02, outer=0.02, speed=0.1, force=20.0)
            # time.sleep(0.1)

            # Move to a higher position after gripping the object 抓起物体向上提
            high_pick_up_pos = self._calculate_approach_position(pick_pos)
            self.robot_mover.grasp_approach(pick_pos, high_pick_up_pos, pick_rpy, retry_init=True)
            # self.robot_mover.move(high_pick_up_pos, pick_rpy, retry_init=True)
            # time.sleep(0.1)

            if extra_pick_patch: # 放置到安全位置，然后从两个方向重新抓

                # 放到安全位置
                safe_position_high = self._calculate_approach_position(SAFE_POSITION)
                self.robot_mover.move(safe_position_high, SAFE_ORIENTATION, retry_init=True)
                # self.robot_mover.move([a + b for a, b in zip(SAFE_POSITION, [0, 0, 0.2])], SAFE_POSITION, retry_init=True)
                self.robot_mover.grasp_approach(safe_position_high, [a + b for a, b in zip(SAFE_POSITION, [0, 0, 0.025])], SAFE_ORIENTATION, retry_init=True ) # 留点位置，防止撞
                time.sleep(0.4)
                self.gripper.open(width=0.08, speed=0.1)
                self.robot_mover.grasp_approach([a + b for a, b in zip(SAFE_POSITION, [0, 0, 0.025])], safe_position_high, SAFE_ORIENTATION, retry_init=True ) # 提起来换方向
                SAFE_NEXT_ORIENTATION = [a + b for a, b in zip(SAFE_ORIENTATION, [0, 0, np.pi/2])]
                self.robot_mover.move(safe_position_high, SAFE_NEXT_ORIENTATION, retry_init=True)
                self.robot_mover.grasp_approach(safe_position_high, SAFE_POSITION, SAFE_NEXT_ORIENTATION, retry_init=True ) # 下去抓起来
                self.gripper.close(width=0.04, inner=0.02, outer=0.02, speed=0.1, force=20.0)
                for i in range(RETRAY_STEPS-1): # 再多试几次
                    self.gripper.open(width=0.08, speed=0.1)
                    self.robot_mover.grasp_approach(SAFE_POSITION,safe_position_high, SAFE_NEXT_ORIENTATION, retry_init=True ) # 提起来
                    SAFE_NEXT_ORIENTATION = [a + b for a, b in zip(SAFE_NEXT_ORIENTATION, [0, 0, np.pi/2])]
                    self.robot_mover.move(safe_position_high, SAFE_NEXT_ORIENTATION, retry_init=True)
                    self.robot_mover.grasp_approach(safe_position_high, SAFE_POSITION, SAFE_NEXT_ORIENTATION, retry_init=True ) # 下去抓起来
                    self.gripper.close(width=0.04, inner=0.02, outer=0.02, speed=0.1, force=20.0)
                self.robot_mover.grasp_approach(SAFE_POSITION, safe_position_high, SAFE_NEXT_ORIENTATION, retry_init=True ) # 下去抓起来
                

            # Move to a higher position for placing the object
            high_place_pos = self._calculate_approach_position(place_pos)
            # self.robot_mover.grasp_approach(high_pick_up_pos, high_place_pos, pick_rpy)
            self.robot_mover.move(high_place_pos, place_rpy, retry_init=True)
            # time.sleep(0.5)

            # Move to the place position
            self.robot_mover.grasp_approach(high_place_pos, place_pos, place_rpy, retry_init=True)
            # self.robot_mover.move(place_pos, place_rpy, retry_init=True)
            time.sleep(0.5)

            # Open the gripper to release the object
            self.gripper.open(width=0.08, speed=0.1)
            # time.sleep(0.5)

            # Return to the higher position after releasing the object 放下物体，回到高位
            self.robot_mover.grasp_approach(place_pos, high_place_pos, place_rpy, retry_init=True)
            # self.robot_mover.move(high_place_pos, place_rpy, retry_init=True)
            # time.sleep(0.5)

        except Exception as e:
            rospy.logerr(f"Error in pick and place: {e}")

    def move(self, position, rpy):
        """Move the robot based on target position and orientation"""
        self.robot_mover.move(position, rpy, retry_init=True)

    def _calculate_approach_position(self, pos):
        """Calculate the approach position by adding the approach distance to the z-coordinate"""
        approach_pos = list(pos)
        approach_pos[2] += self.approach_distance
        return approach_pos


def test():
    rospy.init_node("pick_and_place_node", anonymous=True)

    pick_place = PickAndPlace(approach_distance=0.1)
    pick_pos = [0.4, 0, 0.13]  # Initial position of the object
    pick_rpy = [0, np.pi, np.pi / 2 + np.pi / 4]

    place_pos = [0.5, 0.1, 0.13]  # Target position of the object
    place_rpy = [0, np.pi, np.pi / 2 + np.pi / 4]
    try:
        for i in range(100):
            pick_place.pick_and_place(
                pick_pos=pick_pos,
                pick_rpy=pick_rpy,
                place_pos=place_pos,
                place_rpy=place_rpy,
            )
            pick_pos, place_pos = place_pos, pick_pos

    except rospy.ROSInterruptException:
        pass


def peral():
    # TODO: Grasp Generation, Collision Avoidance by Moveit
    rospy.init_node("pick_and_place_node", anonymous=True)
    pick_place = PickAndPlace(approach_distance=0.3)
    pick_pos_ = [0.61, -0.01, 0.02] 
    pick_rpy = [0.1, np.pi+0.1, np.pi+0.1]
    place_pose = [0.3, 0.0, 0.03] 
    place_rpy = [3.141592653589793, 0.0, -1.5707963267948966]

    pick_place.pick_and_place(
        pick_pos=pick_pos_,
        pick_rpy=pick_rpy,
        place_pos=place_pose,
        place_rpy=place_rpy,
        extra_pick_patch=True
    )

    # pick_rpy = [0, np.pi, np.pi / 4]
    # place_rpy = [0, np.pi, np.pi / 4]

    # # P
    # pick_place.pick_and_place(
    #     pick_pos=[0.6, -0.14, 0.13],
    #     pick_rpy=pick_rpy,
    #     place_pos=[0.40, 0.1, 0.13],
    #     place_rpy=place_rpy,
    # )

    # # E
    # pick_place.pick_and_place(
    #     pick_pos=[0.5, 0.22, 0.13],
    #     pick_rpy=pick_rpy,
    #     place_pos=[0.40, 0.05, 0.13],
    #     place_rpy=place_rpy,
    # )

    # # A
    # pick_place.pick_and_place(
    #     pick_pos=[0.5, -0.02, 0.13],
    #     pick_rpy=pick_rpy,
    #     place_pos=[0.40, -0.05, 0.13],
    #     place_rpy=place_rpy,
    # )

    # # R
    # pick_place.pick_and_place(
    #     pick_pos=[0.6, -0.08, 0.13],
    #     pick_rpy=pick_rpy,
    #     place_pos=[0.40, 0.00, 0.13],
    #     place_rpy=place_rpy,
    # )

    # # L
    # pick_place.pick_and_place(
    #     pick_pos=[0.6, 0.28, 0.13],
    #     pick_rpy=pick_rpy,
    #     place_pos=[0.40, -0.1, 0.13],
    #     place_rpy=place_rpy,
    # )
    # pass


if __name__ == "__main__":
    peral()
