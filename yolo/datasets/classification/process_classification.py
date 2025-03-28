import os
import glob
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import random
import concurrent.futures
import tqdm
import multiprocessing

def process_image_for_classification(img, img_number, rotation):
    import random
    import numpy as np
    from PIL import ImageFilter, ImageEnhance, Image

    # 随机下采样
    downsample_ratio = random.uniform(0.1, 1.0)
    w, h = img.size
    down_w = max(1, int(w * downsample_ratio))
    down_h = max(1, int(h * downsample_ratio))
    img = img.resize((down_w, down_h), Image.BILINEAR)

    # 随机小角度旋转（-10 ~ 10°）
    img = img.rotate(random.uniform(-10, 10), resample=Image.BILINEAR)

    # 基本旋转（四个方向）
    if rotation == 0:
        rotated_img = img
    elif rotation == 1:
        rotated_img = img.transpose(Image.ROTATE_270)
    elif rotation == 2:
        rotated_img = img.transpose(Image.ROTATE_180)
    elif rotation == 3:
        rotated_img = img.transpose(Image.ROTATE_90)

    # ======= 随机仿射变换：拉伸 + 平移 + 扭曲（轻微） =======
    scale_x = random.uniform(0.6, 1.4)  # X轴缩放
    scale_y = random.uniform(0.6, 1.4)  # Y轴缩放
    shear_x = random.uniform(-0.4, 0.4)  # X轴剪切
    shear_y = random.uniform(-0.4, 0.)  # Y轴剪切
    trans_x = random.uniform(-12, 12)   # X轴平移
    trans_y = random.uniform(-12, 12)   # Y轴平移

    affine_matrix = (
        scale_x, shear_x, trans_x,
        shear_y, scale_y, trans_y
    )

    width, height = rotated_img.size
    affine_img = rotated_img.transform(
        (width, height),
        Image.AFFINE,
        data=affine_matrix,
        resample=Image.BILINEAR,
        fill=0
    )

    # 灰度化
    gray_img = affine_img.convert('L')

    # 随机亮度
    enhancer = ImageEnhance.Brightness(gray_img)
    bright_img = enhancer.enhance(random.uniform(0.5, 1.5))

    # 高斯模糊
    blurred_img = bright_img.filter(ImageFilter.GaussianBlur(radius=2))

    # 添加随机黑点噪声（置0）
    img_array = np.array(blurred_img)
    mask = np.random.random(img_array.shape) < 0.4
    img_array[mask] = 0

    # 添加盐粒噪声（置255）
    salt_mask = np.random.random(img_array.shape) < 0.02
    img_array[salt_mask] = 255

    # 转回图像
    noised_img = Image.fromarray(img_array)

    # 上采样至 100x100
    final_img = noised_img.resize((100, 100), Image.BILINEAR)

    return final_img

def process_image(img_path, base_name, rotation):
    with Image.open(img_path) as img:
        processed_img = process_image_for_classification(img, None, rotation)
        target_folder = os.path.join('./train', f'{base_name}_{rotation}')
        existing_files = [f for f in os.listdir(target_folder) if f.endswith('.png')]
        save_name = f"{len(existing_files)}.png"
        save_path = os.path.join(target_folder, save_name)
        processed_img.save(save_path)

def main():
    os.makedirs('./train', exist_ok=True)
    img_paths = glob.glob('./original/*.png')
    num_threads = multiprocessing.cpu_count()

    # 预生成所有任务
    tasks = []
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]

        for r in range(4):
            folder_path = os.path.join('./train', f'{base_name}_{r}')
            os.makedirs(folder_path, exist_ok=True)

        for i in range(100):
            for rotation in range(4):
                tasks.append((img_path, base_name, rotation))

    # 多线程执行所有任务，并显示整体进度
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_image, *task) for task in tasks]
        for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images", unit="task"):
            pass

if __name__ == "__main__":
    main()
