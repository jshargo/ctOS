import cv2
from ultralytics import YOLO


cap = cv2.VideoCapture("dog.mp4")

model = YOLO("yolov8m.pt")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    result = results[0]
    bboxes = result.boxes.xyxy
    print(bboxes)
    
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

