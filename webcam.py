import numpy as np
import time
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    time.sleep(0.25)

cap.release()
cv2.destroyAllWindows()
print("OpenCV version:", cv2.__version__)
