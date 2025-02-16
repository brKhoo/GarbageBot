import serial
import time
import numpy as np
import torch
import time
import cv2
import os
from classify import visualize_model_predictions


bluetooth_port = 'COM7'  
baud_rate = 9600  # Same as Arduino baud rate
bluetooth = serial.Serial(bluetooth_port, baud_rate)
cap = cv2.VideoCapture(0)
#model_ft = torch.load("model_ft.pth")
model_conv = torch.load("model_conv.pth", weights_only=False)

#model_ft.eval()
model_conv.eval()

# Create a directory to save the images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)



def read_data_from_arduino():

    while True:
        if bluetooth.in_waiting > 0:
            incoming_data = bluetooth.readline().decode('utf-8').strip()
            print(f"Data from arduino:{incoming_data}")

            if incoming_data == "Shock Detected":
                classify_trash()

def classify_trash():
    ret, frame = cap.read()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = f"{output_dir}/image_{timestamp}.jpg"
    cv2.imwrite(image_filename, frame)

    print("Running AI model for trash classification...")
    result = visualize_model_predictions(
        model_conv,
        img_path=f"{output_dir}/image_{timestamp}.jpg"
    )
    print(f"Detected trash type: {result}")

    location = 1

    if result == "biological":
        location = 2
    elif result == "metal" or result == "plastic":
        location = 3
    # else #location == "paper":
        # location = 4

    print(f"Sending to location {location}")

    send_result_to_arduino(f"LOCATION{location}")


def send_result_to_arduino(result):
    bluetooth.write(result.encode('utf-8'))
    print(f"Sent to arduino:{result}")


if __name__ == '__main__':
    
    try:
        read_data_from_arduino()
    
    except KeyboardInterrupt:
        print("Program interrupted")
    
    finally:
        bluetooth.close()