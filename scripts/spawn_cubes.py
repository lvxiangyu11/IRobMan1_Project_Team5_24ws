#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose, Quaternion, Point
from tf.transformations import quaternion_from_euler
from gazebo_msgs.srv import SpawnModel
import random
from gazebo_msgs.srv import DeleteModel

cube_sdf="""
<?xml version="1.0" ?>
<sdf version="1.4">
<model name='%NAME%'>
  <static>0</static>
  <link name='%NAME%'>
    <inertial>
        <mass>0.066</mass>
        <inertia> <!-- inertias are tricky to compute -->
          <!-- http://gazebosim.org/tutorials?tut=inertia&cat=build_robot -->
          <ixx>0.0000221859</ixx>       <!-- for a box: ixx = 0.083 * mass * (y*y + z*z) -->
          <ixy>0.0</ixy>         <!-- for a box: ixy = 0 -->
          <ixz>0.0</ixz>         <!-- for a box: ixz = 0 -->
          <iyy>0.0000221859</iyy>       <!-- for a box: iyy = 0.083 * mass * (x*x + z*z) -->
          <iyz>0.0</iyz>         <!-- for a box: iyz = 0 -->
          <izz>0.0000221859</izz>       <!-- for a box: izz = 0.083 * mass * (x*x + y*y) -->
        </inertia>
      </inertial>
    <collision name='collision'>
      <max_contacts>10</max_contacts>
      <surface>
        <contact>
          <ode>
            <max_vel>0</max_vel>
            # <min_depth>0.003</min_depth>
            <min_depth>0.01</min_depth>
          </ode>
        </contact>
        <!--NOTE: Uses dynamic friction of brick on a wood surface
        see https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
        -->
        <friction>
          <ode>
            <mu>10</mu>
            <mu2>10</mu2>
            <fdir1>1 0 0</fdir1>
            <slip1>0</slip1>
            <slip2>0</slip2>
          </ode>
          <torsional>
            <coefficient>1</coefficient>
            <patch_radius>0</patch_radius>
            <surface_radius>0</surface_radius>
            <use_patch_radius>1</use_patch_radius>
            <ode>
              <slip>0</slip>
            </ode>
          </torsional>
        </friction>
        <bounce>
          <restitution_coefficient>0</restitution_coefficient>
          <threshold>1e+06</threshold>
        </bounce>
      </surface>
      <geometry>
        <box>
          <size> 0.045 0.045 0.045 </size>
        </box>
      </geometry>
    </collision>
    <visual name='%NAME%'>
      <pose>0 0 0 0 0 0</pose>
      <geometry>
        <mesh>
          <uri>model://%NAME%.dae</uri>
        </mesh>
      </geometry>
    </visual>
  </link>
  <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
    <frameName>world</frameName>
    <bodyName>%NAME%</bodyName>
    <topicName>%NAME%_odom</topicName>
    <updateRate>30.0</updateRate>
  </plugin>
</model>
"""

cube_urdf="""
<?xml version="1.0" ?>
<robot name="%NAME%" xmlns:xacro="http://ros.org/wiki/xacro">
    <link name="%NAME%">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          # <mass value="0.066" />
          <mass value="1" />
          <inertia ixx="0.0000221859" ixy="0.0" ixz="0.0" iyy="0.0000221859" iyz="0.0" izz="0.0000221859" />
        </inertial>
        <collision>
            <geometry>
              <box size="0.045 0.045 0.045" />
            </geometry>
        </collision>
        <visual>
            <geometry>
              <mesh filename="package://franka_zed_gazebo/meshes/%NAME%.dae" scale='1 1 1'/>
            </geometry>
        </visual>
    </link>
    <gazebo>
      <collision name="%NAME%_collision">
       <max_contacts>10</max_contacts>
        <surface>
          <contact>
            <ode>
              <max_vel>0</max_vel>
              <min_depth>0.003</min_depth>
            </ode>
          </contact>
          <!--NOTE: Uses dynamic friction of brick on a wood surface
          see https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
          -->
          <friction>
            <ode>
              <mu>100000</mu>
              <mu2>100000</mu2>
              <fdir1>0 0 0</fdir1>
              <slip1>0</slip1>
              <slip2>0</slip2>
            </ode>
            <torsional>
              <coefficient>1</coefficient>
              <patch_radius>0</patch_radius>
              <surface_radius>0</surface_radius>
              <use_patch_radius>1</use_patch_radius>
              <ode>
                <slip>0</slip>
              </ode>
            </torsional>
          </friction>
          <bounce>
            <restitution_coefficient>0</restitution_coefficient>
            <threshold>1e+06</threshold>
          </bounce>
        </surface>
      </collision>
      <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
        <frameName>world</frameName>
        <bodyName>%NAME%</bodyName>
        <topicName>%NAME%_odom</topicName>
        <updateRate>30.0</updateRate>
      </plugin>
    </gazebo>
</robot>
"""

rospy.init_node('spawn_cubes', anonymous=True)
Spawning = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel) # you can cange sdf to urdf
rospy.wait_for_service("gazebo/spawn_sdf_model") # you can cange sdf to urdf

def spawn(id, position, orientation):
    """
    删除同名模型后，生成新的模型
    """
    model_name = 'cube_{0}'.format(id)

    # # 尝试删除同名模型
    # try:
    #     Deleting = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    #     rospy.wait_for_service("gazebo/delete_model")
    #     Deleting(model_name)
    #     rospy.loginfo(f"Deleted existing model: {model_name}")
    # except rospy.ServiceException as e:
    #     rospy.logwarn(f"Failed to delete model {model_name}, it might not exist: {e}")

    # 生成新的模型
    model_xml = cube_sdf.replace('%NAME%', model_name)  # 可切换 sdf/urdf
    cube_pose = Pose(Point(*position), Quaternion(*quaternion_from_euler(*orientation)))
    Spawning(model_name, model_xml, "", cube_pose, "world")
    rospy.loginfo("%s was spawned in Gazebo", model_name)

# the ranges for generating cubs
# table size is 0.6 x 0.75
table_xlim=[-0.2,0.2]
table_ylim=[-0.3, 0.3]
table_zlim=[0.1, 0.2]
# table_xlim=[-0.1,0.1]
# table_ylim=[-0.2, 0.2]
# table_zlim=[0.1, 0.2]
# table surface pose
xpose=0.5
ypose=0
zpose=0

for i in range(27):
  model_name = 'cube_{0}'.format(i)
  try:
    Deleting = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    rospy.wait_for_service("gazebo/delete_model")
    Deleting(model_name)
    rospy.loginfo(f"Deleted existing model: {model_name}")
  except rospy.ServiceException as e:
    rospy.logwarn(f"Failed to delete model {model_name}, it might not exist: {e}")

offset = 0.10
for i in range(10):
  print("create_cubes:", i)
  # position=[xpose + random.uniform(*table_xlim),
  #           ypose + random.uniform(*table_ylim),
  #           zpose + random.uniform(*table_zlim)
  # ]
  # orientation=[random.uniform(-1.5,1.5), random.uniform(-1.5,1.5), random.uniform(-1.5,1.5)]
  orientation=[random.uniform(-1.5,1.5), random.uniform(-1.5,1.5), random.uniform(-0.1,0.1)]
  position=[xpose+0.10 ,
            ypose+ i*offset-0.35,
            zpose
  ]
  orientation=[0, 0, -np.pi/4]
  spawn(i, position, orientation)

# for i in range(9):
#   print("create_cubes:", (i+5))
#   # position=[xpose + random.uniform(*table_xlim),
#   #           ypose + random.uniform(*table_ylim),
#   #           zpose + random.uniform(*table_zlim)
#   # ]
#   # orientation=[random.uniform(-1.5,1.5), random.uniform(-1.5,1.5), random.uniform(-1.5,1.5)]
#   position=[xpose +0.05 ,
#             ypose+ i*offset-0.4,
#             zpose
#   ]
#   orientation=[0, 0, -np.pi/2]
#   spawn((i+9), position, orientation)

# for i in range(8):
#   print("create_cubes:", (i+18))
#   # position=[xpose + random.uniform(*table_xlim),
#   #           ypose + random.uniform(*table_ylim),
#   #           zpose + random.uniform(*table_zlim)
#   # ]
#   # orientation=[random.uniform(-1.5,1.5), random.uniform(-1.5,1.5), random.uniform(-1.5,1.5)]
#   position=[xpose ,
#             ypose+ i*offset-0.4,
#             zpose
#   ]
#   orientation=[0, 0, -np.pi/2]
#   spawn((i+18), position, orientation)