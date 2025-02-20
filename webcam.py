import numpy as np
import torch
import time
import cv2
import os
from classify import visualize_model_predictions

cap = cv2.VideoCapture(0)

#if running on a machine with cuda-capable gpu, remove map_location=torch.device('cpu'), as this forces the machine to use cpu
model_ft = torch.load("model_ft.pth", weights_only=False, map_location=torch.device('cpu'))
model_conv = torch.load("model_conv.pth", weights_only=False, map_location=torch.device('cpu'))

model_ft.eval()
model_conv.eval()
        
# # Create a directory to save the images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# # Start the loop to capture frames every 3 seconds
while True:
    #if bluetooth signal detected: --------------------------------------------------------------------

    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Get the current timestamp for naming the image
    #timestamp = time.strftime("%Y%m%d_%H%M%S")
    #image_filename = f"{output_dir}/image_{timestamp}.jpg"
        
    # Save the image
    #cv2.imwrite(image_filename, frame)

    #Classify the image, and display it in an openc window
    text = visualize_model_predictions(model_ft,#img_path=f"{output_dir}/image_{timestamp}.jpg"
                                       frame)
    coordinates = (50, 50)
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2
    frame = cv2.putText(frame, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Classification", frame)
    #send out the classification of the image to the bluetooth receiver --------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #os.remove(image_filename)

    #print(f"Captured image: {image_filename}")
    #time.sleep(3)

# Release the webcam and close any OpenCV windows
#os.remove(image_filename)
cap.release()
cv2.destroyAllWindows()
