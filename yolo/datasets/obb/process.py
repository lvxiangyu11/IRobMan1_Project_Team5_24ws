import os
import random
import glob
import yaml
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
import threading

lock = threading.Lock()


def generate_img_obb(img, canvas_size=(640, 360)):
    while True:
        orig_size = img.size
        scale_factor = random.uniform(0.05, 0.15)
        rotation_angle = random.uniform(-180, 180)
        dx = random.randint(-200, 200)
        dy = random.randint(-80, 80)

        # 缩放处理
        scaled_size = (int(orig_size[0] * scale_factor),
                       int(orig_size[1] * scale_factor))
        img_scaled = img.resize(scaled_size, Image.Resampling.LANCZOS)

        # 仿射变换（示例：随机剪切）
        shear_x = random.uniform(-0.1, 0.1)
        shear_y = random.uniform(-0.1, 0.1)
        affine_matrix = (1, shear_x, 0, shear_y, 1, 0)
        img_scaled = img_scaled.transform(
            img_scaled.size,
            Image.AFFINE,
            data=affine_matrix,
            resample=Image.Resampling.BICUBIC
        )

        # 创建透明画布
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        # 计算粘贴位置（使用用户指定的偏移量）
        paste_pos = (
            (canvas_size[0] - scaled_size[0]) // 2 + dx,  # 应用水平偏移
            (canvas_size[1] - scaled_size[1]) // 2 + dy   # 应用垂直偏移
        )

        # 旋转图像（以图像中心为旋转中心）
        img_rotated = img_scaled.rotate(
            rotation_angle,
            expand=True,
            resample=Image.BICUBIC,
            center=(scaled_size[0] / 2, scaled_size[1] / 2)
        )

        # 将旋转后的图像粘贴到画布
        canvas.paste(img_rotated, paste_pos, img_rotated)

        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGRA)

        # 基于Alpha通道的轮廓检测
        alpha = cv_image[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue  # 重新尝试生成

        # 获取最大轮廓的OBB
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.float32)

        # 检查 OBB 是否超出 Canvas 范围
        if all(0 <= x <= canvas_size[0] and 0 <= y <= canvas_size[1] for x, y in box):
            # 坐标归一化
            normalized = []
            for x, y in box:
                nx = x / canvas_size[0]
                ny = y / canvas_size[1]
                normalized.extend([nx, ny])
            return cv_image, normalized


def add_noise(image):
    """给图像添加多种噪声"""
    # 将PIL图像转换为OpenCV格式（RGB）
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 1. 盐粒噪声 (Salt and Pepper)
    s_vs_p = 0.5
    amount = 0.004
    noisy = np.copy(img_cv)
    # 添加盐噪声
    num_salt = np.ceil(amount * img_cv.shape[0] * img_cv.shape[1] * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_cv.shape[:2]]
    noisy[coords[0], coords[1], :] = 255
    # 添加胡椒噪声
    num_pepper = np.ceil(amount * img_cv.shape[0] * img_cv.shape[1] * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_cv.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    # 2. 高斯噪声
    mean = 0
    sigma = 15  # 调整此参数以控制高斯噪声的强度
    gaussian_noise = np.random.normal(mean, sigma, noisy.shape).astype(np.float32)
    noisy = np.clip(noisy.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)

    # 3. 添加一些随机亮度和对比度调整
    alpha = random.uniform(0.9, 1.1)  # 对比度
    beta = random.randint(-10, 10)    # 亮度
    noisy = cv2.convertScaleAbs(noisy, alpha=alpha, beta=beta)

    # 4. 添加一些模糊
    if random.random() < 0.3:  # 30%的概率添加模糊
        blur_size = random.choice([3, 5])
        noisy = cv2.GaussianBlur(noisy, (blur_size, blur_size), 0)

    # 转回PIL格式
    return Image.fromarray(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))


def prepare_yolo_dataset_multithread(
    original_dir='./original',
    train_dir='./train',
    test_dir='./test',
    cfg_dir='./cfg',
    k=5,
    test_ratio=0.1 # 设置测试集比例，比如20%
):
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(original_dir, '*.png'))
    train_txt_path = os.path.join(cfg_dir, 'train.txt')
    test_txt_path = os.path.join(cfg_dir, 'test.txt')
    classes_txt_path = os.path.join(cfg_dir, 'classes.txt')
    obj_data_path = os.path.join(cfg_dir, 'obj.data')
    yaml_path = os.path.join(cfg_dir, 'dataset.yaml')

    class_names = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in image_files))
    with open(classes_txt_path, 'w') as f:
        f.write('\n'.join(class_names))

    with open(obj_data_path, 'w') as f:
        f.write(f"classes = {len(class_names)}\n")
        f.write(f"train = {train_txt_path}\n")
        f.write(f"valid = {test_txt_path}\n")
        f.write(f"names = {classes_txt_path}\n")
        f.write("backup = backup/")

    dataset_yaml = {
        'train': train_txt_path,
        'valid': test_txt_path,
        'nc': len(class_names),
        'names': class_names
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    canvas_size = (640, 360)
    results_train, results_test = [], []

    def worker(i):
        selected_imgs = random.sample(image_files, random.randint(3, 6))
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        label_data, placed_contours = [], []

        for img_path in selected_imgs:
            class_name = os.path.splitext(os.path.basename(img_path))[0]
            class_index = class_names.index(class_name)
            img = Image.open(img_path).convert("RGBA")

            while True:
                cv_image, normalized = generate_img_obb(img, canvas_size)
                current_contour = np.array(
                    [(int(normalized[j] * canvas_size[0]), int(normalized[j + 1] * canvas_size[1]))
                     for j in range(0, 8, 2)], dtype=np.int32)

                overlap = any(
                    cv2.rotatedRectangleIntersection(cv2.minAreaRect(pc), cv2.minAreaRect(current_contour))[0] > 0
                    for pc in placed_contours
                )

                if not overlap:
                    placed_contours.append(current_contour)
                    break

            canvas.paste(Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)), (0, 0),
                         mask=Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)))

            label_data.append(f"{class_index} {' '.join([f'{c:.6f}' for c in normalized])}")

        canvas = add_noise(canvas.convert("RGB"))
        merged_filename = f"{i + 1}.png"

        # 根据test_ratio划分训练集和测试集
        if random.random() < test_ratio:
            current_img_path = os.path.join(test_dir, 'images', merged_filename)
            current_label_path = os.path.join(test_dir, 'labels', f"{i + 1}.txt")
            with lock:
                results_test.append(current_img_path)
        else:
            current_img_path = os.path.join(train_dir, 'images', merged_filename)
            current_label_path = os.path.join(train_dir, 'labels', f"{i + 1}.txt")
            with lock:
                results_train.append(current_img_path)

        canvas.save(current_img_path)
        with open(current_label_path, 'w') as f:
            f.write("\n".join(label_data))

    num_threads = multiprocessing.cpu_count()*1.2
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(worker, range(k)), total=k, desc="生成数据集"))

    # 写入train.txt 和 test.txt
    with open(train_txt_path, 'w') as train_file:
        train_file.write("\n".join(sorted(results_train, key=lambda x: int(os.path.basename(x).split('.')[0]))))

    with open(test_txt_path, 'w') as test_file:
        test_file.write("\n".join(sorted(results_test, key=lambda x: int(os.path.basename(x).split('.')[0]))))

    print(f"数据集生成完成！共生成 {k} 张合成图像，其中训练集 {len(results_train)} 张，测试集 {len(results_test)} 张。")

def visualize_obb(img_path, all_coords):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 遍历多个 OBB 并绘制
    for coords in all_coords:
        points = [(int(coords[i] * w), int(coords[i + 1] * h)) for i in range(0, 8, 2)]
        points = np.array(points, dtype=np.int32)

        # 绘制 OBB
        cv2.polylines(img, [points], True, (0, 255, 0), 2)

        # 显示坐标信息
        for x, y in points:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(img, f"{x},{y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    k = 50000  # 可以通过修改此参数改变生成的图像数量
    prepare_yolo_dataset_multithread(k=k)