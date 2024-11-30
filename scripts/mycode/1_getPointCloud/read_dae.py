import collada
import numpy as np
import open3d as o3d

# 指定文件路径
file_path = "/opt/ros_ws/src/franka_zed_gazebo/meshes/cube_0.dae"

try:
    # 加载 .dae 文件
    dae_mesh = collada.Collada(file_path)
    
    # 提取几何信息
    for geometry in dae_mesh.geometries:
        for primitive in geometry.primitives:
            if isinstance(primitive, collada.triangleset.TriangleSet):
                # 获取顶点和索引
                vertices = np.array(primitive.vertex).astype(np.float32)
                faces = np.array(primitive.indices).astype(np.int32)

                # 创建 Open3D 的 TriangleMesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()

                # 显示网格
                o3d.visualization.draw_geometries([mesh], window_name="3D Mesh Viewer")
except Exception as e:
    print(f"Failed to load or display the file: {file_path}")
    print(f"Error: {e}")
