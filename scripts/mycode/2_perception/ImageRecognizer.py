import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import easyocr

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
    test_image_np = np.array(Image.open("/opt/ros_ws/src/franka_zed_gazebo/scripts/mycode/2_perception/test.png"))

    # 识别图像并显示结果
    results = recognizer.recognize_image(test_image_np)
    print(results[0])

    # 获取并显示第一个结果的图像
    image_from_result = recognizer.get_image_from_result(results[0])
    # plt.imshow(image_from_result)
    # plt.title(f"Result Image: {results[0][2]} at {results[0][3]}°")
    # plt.axis("off")
    # plt.show()

    recognizer.display_results(test_image_np, results[:10])

    # 使用 CNN 和文字筛选最佳匹配
    best_result = recognizer.selected_best_based_on_CNN(results, test_image_np)
    print("Best Result:", best_result)
    img, angle = recognizer.get_image_from_result(best_result, True)
    print(angle)