import os
import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO("./runs/classify/train7   /weights/best.pt")

def predict_image(image_path):
    """读取图像，处理后进行预测，返回一个 list，包含按置信度降序排列的 (类名, 置信度) 元组"""
    img = cv2.imread(image_path)
    if img is None:
        return []

    # 灰度处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 缩放到100x100
    resized_img = cv2.resize(gray_img, (100, 100))

    # 转换为3通道
    resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

    # 模型预测
    results = model(resized_img_rgb)
    result = results[0]

    # 获取预测概率
    probs = result.probs
    top5 = probs.top5

    # 构建返回列表
    predictions = []
    for idx in top5:
        class_name = result.names[idx]
        confidence = float(probs.data[idx])
        predictions.append((class_name, confidence))

    # 按置信度降序排序
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions

# 遍历所有图像文件
image_dir = "./datasets/classification/real/"
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(image_dir, filename)
        preds = predict_image(path)
        print(f"\n{filename} Predictions:")
        for cls, conf in preds:
            print(f" - {cls}: {conf:.4f}")
