from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")
results = model_yolo("./images/soccer.jpg")
results.show()