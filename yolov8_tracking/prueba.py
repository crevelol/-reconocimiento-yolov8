from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')

model.train(data='coco128.yaml', epochs=100, imgsz=640)