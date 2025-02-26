#!/usr/bin/env python

import numpy as np
import rospy
from mygripper import MyGripper
from mymove import MoveRobot
import time


class PickAndPlace:
    def __init__(self, approach_distance=0.15):
        self.robot_mover = MoveRobot()
        self.gripper = MyGripper()
        self.approach_distance = approach_distance

    def pick_and_place(
        self, pick_pos, pick_rpy, place_pos, place_rpy  # Current position of the object
    ):  # Target position of the object
        try:
            # Open the gripper
            self.gripper.open(width=0.08, speed=0.1)
            time.sleep(1)

            # Move to a higher position above the object
            high_pick_pos = self._calculate_approach_position(pick_pos)
            self.robot_mover.move(high_pick_pos, pick_rpy, 0.2)
            time.sleep(0.2)

            # Move to the object's position
            # self.robot_mover.grasp_approach(high_pick_pos, pick_pos, pick_rpy)
            self.robot_mover.move(pick_pos, pick_rpy, 0.2)
            time.sleep(0.3)

            # Close the gripper
            self.gripper.close(width=0.04, inner=0.02, outer=0.02, speed=0.1, force=1.0)
            time.sleep(0.3)

            # Move to a higher position after gripping the object
            high_pick_up_pos = self._calculate_approach_position(pick_pos)
            # self.robot_mover.grasp_approach(pick_pos, high_pick_up_pos, pick_rpy)
            self.robot_mover.move(high_pick_up_pos, pick_rpy)
            time.sleep(0.2)

            # Move to a higher position for placing the object
            high_place_pos = self._calculate_approach_position(place_pos)
            # self.robot_mover.grasp_approach(high_pick_up_pos, high_place_pos, pick_rpy)
            self.robot_mover.move(high_place_pos, place_rpy, 0.2)
            time.sleep(0.5)

            # Move to the place position
            # self.robot_mover.grasp_approach(high_place_pos, place_pos, place_rpy)
            self.robot_mover.move(place_pos, place_rpy)
            time.sleep(0.5)

            # Open the gripper to release the object
            self.gripper.open(width=0.07, speed=0.1)
            time.sleep(0.5)

            # Return to the higher position after releasing the object
            # self.robot_mover.grasp_approach(place_pos, high_place_pos, place_rpy)
            self.robot_mover.move(high_place_pos, place_rpy)
            time.sleep(0.5)

        except Exception as e:
            rospy.logerr(f"Error in pick and place: {e}")

    def move(self, position, rpy):
        """Move the robot based on target position and orientation"""
        self.robot_mover.move(position, rpy)

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
    pick_place = PickAndPlace(approach_distance=0.1)
    pick_rpy = [0, np.pi, np.pi / 4]
    place_rpy = [0, np.pi, np.pi / 4]

    # P
    pick_place.pick_and_place(
        pick_pos=[0.6, -0.14, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, 0.1, 0.13],
        place_rpy=place_rpy,
    )

    # E
    pick_place.pick_and_place(
        pick_pos=[0.5, 0.22, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, 0.05, 0.13],
        place_rpy=place_rpy,
    )

    # A
    pick_place.pick_and_place(
        pick_pos=[0.5, -0.02, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, -0.05, 0.13],
        place_rpy=place_rpy,
    )

    # R
    pick_place.pick_and_place(
        pick_pos=[0.6, -0.08, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, 0.00, 0.13],
        place_rpy=place_rpy,
    )

    # L
    pick_place.pick_and_place(
        pick_pos=[0.6, 0.28, 0.13],
        pick_rpy=pick_rpy,
        place_pos=[0.40, -0.1, 0.13],
        place_rpy=place_rpy,
    )
    pass


if __name__ == "__main__":
    peral()
