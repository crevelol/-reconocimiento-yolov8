from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')
results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=2,
   batch=8,
   name='yolov8n_custom')

results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='-')
print(success)