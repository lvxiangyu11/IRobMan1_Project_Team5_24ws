import open3d as o3d

# 指定 .obj 文件路径
file_path = "/opt/ros_ws/src/franka_zed_gazebo/meshes/cube_0.obj"

# 加载 .obj 文件
mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=True)

# 检查是否加载了纹理或顶点颜色
if mesh.has_vertex_colors():
    print("Mesh loaded with vertex colors.")
elif mesh.has_triangle_uvs():
    print("Mesh loaded with texture coordinates (UVs).")
    print("Note: Check if the associated texture image is correctly loaded.")
else:
    print("Warning: No vertex colors or texture coordinates found.")

# 可视化
o3d.visualization.draw_geometries([mesh])
