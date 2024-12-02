import bpy
import os
import math

# 文件路径设置
input_dir = r"C:\workspace\irobman1_project\meshes"
output_base_dir = r"C:\workspace\irobman1_project\tmp\cubes"

# 确保输出目录存在
os.makedirs(output_base_dir, exist_ok=True)

# 删除场景中的网格对象和摄像机
def delete_objects():
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'CAMERA'}:
            bpy.data.objects.remove(obj, do_unlink=True)
    print("Deleted existing mesh objects and cameras.")

# 加载 .dae 文件
def import_dae(filepath):
    try:
        bpy.ops.wm.collada_import(filepath=filepath)
        print(f"Successfully imported: {filepath}")
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                return obj
        print("No mesh objects found in the imported file.")
        return None
    except Exception as e:
        print(f"Failed to import {filepath}: {e}")
        return None

# 创建摄像机并设置视角
def setup_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'PERSP'
    camera.data.clip_start = 0.001
    bpy.context.scene.camera = camera
    return camera

# 设置环境光（全局光照）
def setup_environment_lighting(strength=1.0):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in nodes:
        nodes.remove(node)
    background = nodes.new(type='ShaderNodeBackground')
    background.inputs['Strength'].default_value = strength
    output = nodes.new(type='ShaderNodeOutputWorld')
    links.new(background.outputs['Background'], output.inputs['Surface'])

# 设置渲染参数
def setup_render(output_path):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.resolution_percentage = 100

# 设置摄像机位置和拍摄
def render_from_angle(camera, location, rotation, output_path):
    camera.location = location
    camera.rotation_euler = rotation
    setup_render(output_path)
    bpy.ops.render.render(write_still=True)

# 主逻辑
def process_all_cubes(light_strength=2.0, length=0.085):
    # 遍历目录中所有的 cube_?.dae 文件
    for filename in os.listdir(input_dir):
        if filename.startswith("cube_") and filename.endswith(".dae"):
            file_id = filename.split('_')[1].split('.')[0]  # 获取文件编号
            input_file = os.path.join(input_dir, filename)
            output_dir = os.path.join(output_base_dir, file_id)
            os.makedirs(output_dir, exist_ok=True)

            delete_objects()
            obj = import_dae(input_file)
            if obj is None:
                continue

            setup_environment_lighting(strength=light_strength)
            camera = setup_camera()

            views = {
                "front": ((0, -length, 0), (math.radians(90), 0, 0)),
                "back": ((0, length, 0), (math.radians(90), 0, math.radians(180))),
                "left": ((length, 0, 0), (math.radians(90), 0, math.radians(90))),
                "right": ((-length, 0, 0), (math.radians(90), 0, math.radians(-90))),
                "top": ((0, 0, length), (0, 0, 0)),
                "bottom": ((0, 0, -length), (math.radians(180), 0, 0)),
            }

            for view_name, (location, rotation) in views.items():
                output_path = os.path.join(output_dir, f"{view_name}.png")
                render_from_angle(camera, location, rotation, output_path)

            print(f"Processed {filename}, saved images in {output_dir}")

# 执行脚本
process_all_cubes(light_strength=1)
