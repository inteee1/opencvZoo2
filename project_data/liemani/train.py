from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on COCO8
results = model.train(data="/home/inteee/opencvZoo2/project_data/liemani/ipad2_training.yaml", epochs=100, imgsz=320)