import cv2
import numpy as np
from ultralytics import YOLO

device = "cpu"
cap = cv2.VideoCapture("fields.mp4")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame, device=device)
    
    result = results[0]
    
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)  # Red bounding box
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  # Class label
    

    cv2.imshow("YOLOv8 Object Detection", frame)
    

cap.release()
cv2.destroyAllWindows()
