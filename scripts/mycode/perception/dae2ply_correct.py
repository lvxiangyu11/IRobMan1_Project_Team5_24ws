import os
from PIL import Image

# 定义目录路径
base_dir = r"tmp\cubes"

# 遍历目录
for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)

        # 如果是 back.png 文件，镜像翻转
        if file == "back.png":
            with Image.open(file_path) as img:
                mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mirrored_img.save(file_path)  # 保存处理后的图片
                print(f"Processed {file_path}: mirrored")

        # 如果是 left.png 文件，逆时针旋转90度
        elif file == "left.png":
            with Image.open(file_path) as img:
                rotated_img = img.rotate(90, expand=True)
                rotated_img.save(file_path)  # 保存处理后的图片
                print(f"Processed {file_path}: rotated 90 degrees counterclockwise")

        # 如果是 bottom.png 文件，旋转180度再镜像翻转
        elif file == "bottom.png":
            with Image.open(file_path) as img:
                rotated_img = img.rotate(180, expand=True)
                mirrored_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
                mirrored_img.save(file_path)  # 保存处理后的图片
                print(f"Processed {file_path}: rotated 180 degrees and mirrored")
