from ultralytics import YOLO
import easyocr

model = YOLO('yolov8n.pt')
print("YOLOv8 Nano loaded successfully locally")

reader = easyocr.Reader(['en'])
print("EasyOCR loaded successfully locally")

print("Local environment fully ready!")