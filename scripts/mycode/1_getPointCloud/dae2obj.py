import pyassimp
import open3d as o3d

dae_file_path = "/opt/ros_ws/src/franka_zed_gazebo/meshes/cube_0.dae"
obj_file_path = "/opt/ros_ws/src/franka_zed_gazebo/meshes/cube_0.obj"

# Load the DAE file
try:
    scene = pyassimp.load(dae_file_path)
    print("DAE file loaded successfully!")
    
    # Export to OBJ format
    pyassimp.export(scene, obj_file_path, file_type="obj")
    print(f"File converted to OBJ format: {obj_file_path}")
    
    # Read converted OBJ with Open3D
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    o3d.visualization.draw_geometries([mesh])
except Exception as e:
    print(f"Error processing the DAE file: {e}")
finally:
    pyassimp.release(scene)
