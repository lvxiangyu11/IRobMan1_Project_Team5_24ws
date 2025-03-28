from ultralytics import YOLO

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolo11m-obb.pt")

# Train the model on the DOTAv1 dataset
results = model.train(data="./datasets/dataset.yaml", epochs=100, imgsz=640)