import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import easyocr
import cv2

class ImageRecognizer:
    def __init__(self, top_dir="cubes/"):
        self.top_dir = top_dir
        self.cubes_images = {}
        self.rotated_test_images = {}
        self._load_images()
        
    def _load_images(self):
        """加载文件夹中的所有图片"""
        first_right = True
        for i, folder in enumerate(sorted(os.listdir(self.top_dir))):
            folder_path = os.path.join(self.top_dir, folder)
            if os.path.isdir(folder_path):
                self.cubes_images[i] = {}
                for file in sorted(os.listdir(folder_path)):
                    # if file.endswith(".png"):
                    if file.endswith(".png") :
                        if ("right" in file.lower()):
                            if first_right:
                                first_right = False
                                # read one right 
                            else:
                                continue
                        
                        file_name = os.path.splitext(file)[0]
                        file_path = os.path.join(folder_path, file)
                        image = Image.open(file_path)
                        self.cubes_images[i][file_name] = image

    def rotate_image(self, image, angles=[0, 90, 180, 270]):
        """对输入的图像进行旋转"""
        rotated_images = {}
        for angle in angles:
            rotated_images[angle] = np.array(image.rotate(angle, resample=Image.Resampling.BICUBIC))
        return rotated_images

    def recognize_image(self, test_image_np):
        """识别输入图像，返回最相似的图片的相关信息"""
        test_image = Image.fromarray(test_image_np)
        rotated_test_images = self.rotate_image(test_image)

        similarity_results = []

        for i in self.cubes_images:
            for j in self.cubes_images[i]:
                current_image = np.array(self.cubes_images[i][j])
                height, width, _ = test_image_np.shape
                current_image_resized = np.array(
                    Image.fromarray(current_image).resize((width, height), Image.Resampling.LANCZOS)
                )

                best_similarity = -1
                best_angle = 0

                for angle, rotated_test_image in rotated_test_images.items():
                    total_similarity = 0
                    for c in range(3):  # 对 RGB 三个通道分别计算
                        similarity, _ = ssim(current_image_resized[:, :, c], rotated_test_image[:, :, c], full=True)
                        total_similarity += similarity

                    average_similarity = total_similarity / 3
                    if average_similarity > best_similarity:
                        best_similarity = average_similarity
                        best_angle = angle

                similarity_results.append((best_similarity, i, j, best_angle))

        # similarities = [result[0] for result in similarity_results]
        # dynamic_threshold = np.percentile(similarities, 90)

        # significant_results = [result for result in similarity_results if result[0] > dynamic_threshold]
        # significant_results.sort(key=lambda x: x[0], reverse=True)

        # 对结果按相似度降序排序
        similarity_results.sort(key=lambda x: x[0], reverse=True)


        return similarity_results[:10]
    
    def match_features_with_orb(self, test_image_np):
        """
        使用特征点匹配（ORB）算法对输入彩色图像进行识别，返回最相似的图片的相关信息。
        """
        # 将输入图像转换为 OpenCV 格式（BGR）
        # test_image = cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR)
        test_image = test_image_np
        # 创建 ORB 特征检测器（调整参数）
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

        # 计算输入测试图像的特征点和描述子
        keypoints_test, descriptors_test = orb.detectAndCompute(test_image, None)

        if descriptors_test is None:
            raise ValueError("No descriptors found for the test image.")

        # 匹配结果存储
        matching_results = []

        # 遍历数据库中的每张图片
        for i in self.cubes_images:
            for j in self.cubes_images[i]:
                # 获取当前图像并调整大小以匹配输入图像
                current_image_np = np.array(self.cubes_images[i][j])
                if current_image_np.shape[2] == 4:  # 检测到 4 通道
                    current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_RGBA2BGR)
                if len(current_image_np.shape) != 3 or current_image_np.shape[2] != 3:
                    print(f"Invalid image format for {i}-{j}. Skipping.")
                    continue

                current_image_resized = cv2.resize(
                    current_image_np, 
                    (test_image_np.shape[1], test_image_np.shape[0]), 
                    interpolation=cv2.INTER_AREA
                )

                # 图像增强
                enhanced_image = cv2.equalizeHist(cv2.cvtColor(current_image_resized, cv2.COLOR_BGR2GRAY))
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

                # 计算当前图像的特征点和描述子
                keypoints_current, descriptors_current = orb.detectAndCompute(enhanced_image, None)

                if descriptors_current is None:
                    print(f"No descriptors found for image {i}-{j}. Skipping.")
                    continue

                # 特征点匹配
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(descriptors_test, descriptors_current)

                # 根据距离排序匹配结果，取前 50 个最优匹配点
                matches = sorted(matches, key=lambda x: x.distance)[:50]

                # 计算匹配分数（匹配点越多，分数越高，距离越短，分数越高）
                match_score = sum([1 / (match.distance + 1e-5) for match in matches])

                # 保存匹配结果
                matching_results.append((match_score, i, j, len(matches)))

        # 对结果按匹配分数降序排序
        matching_results.sort(key=lambda x: x[0], reverse=True)

        return matching_results[:10]
    

    def match_features_with_sift(self, test_image_np):
        """
        使用 SIFT 特征点匹配算法对输入彩色图像进行识别，返回最相似的图片的相关信息。
        """
        # 将输入图像直接作为测试图像
        test_image = test_image_np

        # 创建 SIFT 特征检测器（调整参数）
        sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=20, sigma=1.0)

        # 提取测试图像的特征点和描述子
        keypoints_test, descriptors_test = sift.detectAndCompute(test_image, None)

        if descriptors_test is None:
            raise ValueError("No descriptors found for the test image.")

        # 匹配结果存储
        matching_results = []

        # 遍历数据库中的每张图片
        for i in self.cubes_images:
            for j in self.cubes_images[i]:
                # 获取当前图像并调整大小以匹配输入图像
                current_image = np.array(self.cubes_images[i][j])

                # 检查通道数并转换为 3 通道（必要时）
                if current_image.shape[2] == 4:  # RGBA 转换为 BGR
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGBA2BGR)

                if len(current_image.shape) != 3 or current_image.shape[2] != 3:
                    print(f"Invalid image format for {i}-{j}. Skipping.")
                    continue

                current_image_resized = cv2.resize(
                    current_image,
                    (test_image.shape[1], test_image.shape[0]),
                    interpolation=cv2.INTER_AREA
                )

                # 图像增强：对比度增强或锐化（根据需要选择是否开启）
                gray = cv2.cvtColor(current_image_resized, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(gray)
                enhanced_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

                # 提取当前图像的特征点和描述子
                keypoints_current, descriptors_current = sift.detectAndCompute(enhanced_image, None)

                if descriptors_current is None:
                    print(f"No descriptors found for image {i}-{j}. Skipping.")
                    continue

                # 特征点匹配
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT 使用 L2 距离
                matches = bf.match(descriptors_test, descriptors_current)

                # 根据距离排序匹配结果，取前 50 个最优匹配点
                matches = sorted(matches, key=lambda x: x.distance)[:50]

                # 计算匹配分数（匹配点越多，分数越高，距离越短，分数越高）
                match_score = sum([1 / (match.distance + 1e-5) for match in matches])

                # 保存匹配结果
                matching_results.append((match_score, i, j, len(matches)))

        # 对结果按匹配分数降序排序
        matching_results.sort(key=lambda x: x[0], reverse=True)

        return matching_results[:10]

    def get_image_from_result(self, result, show=False):
        """
        从结果中提取并返回对应的图像（包括旋转后的图像）。
        :param result: 一个包含 (score, folder, image_name, angle) 的元组。
        :return: 返回与识别结果相关的图像 (numpy array)，包括旋转的弧度角度。
        """
        score, folder, image_name, angle = result
        
        # 获取原始图像
        original_image = self.cubes_images[folder][image_name]
        
        # 将角度转换为弧度
        angle_radians = angle * np.pi / 180.0

        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(original_image)
            plt.title(f"Image: {image_name}\nAngle: {angle_radians:.2f} radians")
            plt.axis("off")
            plt.show()        

        # 返回图像和弧度角度
        return np.array(original_image), angle_radians

    def display_results(self, test_image_np, results, max_display=10):
        """可视化识别结果"""
        width, height, _ = test_image_np.shape
        plt.figure(figsize=(15, 6))

        # 显示 test_image
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(test_image_np)
        plt.title("Test Image")
        plt.axis("off")

        # 显示显著性图片
        for idx, (score, folder, image_name, angle) in enumerate(results[:max_display]):
            current_image = np.array(self.cubes_images[folder][image_name])
            current_image_resized = np.array(
                Image.fromarray(current_image).resize((width, height), Image.Resampling.LANCZOS)
            )

            plt.subplot(1, len(results) + 1, idx + 2)
            plt.imshow(current_image_resized)
            plt.title(f"Folder {folder}\nImage {image_name}\nAngle {angle}°\nSim: {score:.4f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def selected_best_based_on_CNN(self, results, test_image_np):
        """
        从识别结果中选出最佳匹配的一个。
        :param results: 来自 recognize_image() 方法的结果列表
        :param test_image_np: 测试图片的 numpy 数组
        :return: 最终匹配的结果 (相似度, 文件夹编号, 文件名, 旋转角度)
        """
        # 初始化 EasyOCR
        reader = easyocr.Reader(['en'], gpu=True)

        # 将测试图像转换为 PIL 图像
        test_image = Image.fromarray(test_image_np)

        # 初始化最高相似度的结果
        best_result = results[0]

        for result in results[:10]:  # 仅尝试前 10 个结果
            _, folder, image_name, angle = result

            # 获取原始图像并按角度旋转
            original_image = self.cubes_images[folder][image_name]
            rotated_image = original_image.rotate(angle, resample=Image.Resampling.BICUBIC)
            rotated_image_np = np.array(rotated_image)

            # 对测试图像按角度旋转
            test_rotated = test_image.rotate(angle, resample=Image.Resampling.BICUBIC)
            test_rotated_np = np.array(test_rotated)

            # OCR 提取文本
            test_text = reader.readtext(test_rotated_np, detail=0)
            recognized_text = reader.readtext(rotated_image_np, detail=0)
            # print(test_text, recognized_text)

            if not test_text or not recognized_text:
                continue  # 如果 OCR 结果为空，跳过

            # 转换为字符串
            test_text_str = " ".join(test_text)
            recognized_text_str = " ".join(recognized_text)

            # 判断是否包含英文字符
            if not test_text_str.isalpha() or not recognized_text_str.isalpha():
                continue

            # 如果测试图像的文字和当前候选图像的文字匹配，则返回此结果
            if test_text_str.strip().lower() == recognized_text_str.strip().lower():
                # print("find it !")
                return result

        # 如果没有任何匹配，返回最初相似度最高的结果
        return best_result


if __name__ == "__main__":
    # 示例用法
    recognizer = ImageRecognizer(top_dir="/opt/ros_ws/src/franka_zed_gazebo/scripts/mycode/2_perception/cubes/")
    test_image_np = np.array(Image.open("/opt/ros_ws/src/franka_zed_gazebo/scripts/mycode/2_perception/test_0.png"))

    # 识别图像并显示结果
    results = recognizer.recognize_image(test_image_np)
    print(results[0])
    recognizer.display_results(test_image_np, results[:10])

    results = recognizer.match_features_with_orb(test_image_np)
    print(results[0])
    recognizer.display_results(test_image_np, results[:10])

    results = recognizer.match_features_with_sift(test_image_np)
    print(results[0])

    # 获取并显示第一个结果的图像
    image_from_result = recognizer.get_image_from_result(results[0])
    # plt.imshow(image_from_result)
    # plt.title(f"Result Image: {results[0][2]} at {results[0][3]}°")
    # plt.axis("off")
    # plt.show()

    recognizer.display_results(test_image_np, results[:10])

    # 使用 CNN 和文字筛选最佳匹配
    # best_result = recognizer.selected_best_based_on_CNN(results, test_image_np)
    # print("Best Result:", best_result)
    # img, angle = recognizer.get_image_from_result(best_result, True)
    # print(angle)