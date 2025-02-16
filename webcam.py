import numpy as np
import torch
import time
import cv2
import os
from classify import visualize_model_predictions

cap = cv2.VideoCapture(0)
#model_ft = torch.load("model_ft.pth")
model_conv = torch.load("model_conv.pth")

#model_ft.eval()
model_conv.eval()

# Create a directory to save the images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# Start the loop to capture frames every 3 seconds
while True:
    #if bluetooth signal detected: --------------------------------------------------------------------

    # Capture a frame from the webcam
    ret, frame = cap.read()

    # If the frame is captured correctly
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get the current timestamp for naming the image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = f"{output_dir}/image_{timestamp}.jpg"
        
    # Save the image
    cv2.imwrite(image_filename, frame)

    #send out the classification of the image somehow to the bluetooth receiver --------------------------------------------------
    print(visualize_model_predictions(
    model_conv,
    img_path=f"{output_dir}/image_{timestamp}.jpg"
    ))

    #print(f"Captured image: {image_filename}")
    #time.sleep(3)

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()