from ultralytics import YOLO

if __name__ == '__main__':
    # Create a new YOLO11n-OBB model from scratch
    model = YOLO("yolo11m-cls.pt")

    # Train the model on the DOTAv1 dataset
    results = model.train(data="./datasets/classification", epochs=25, imgsz=100)