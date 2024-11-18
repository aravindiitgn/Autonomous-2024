from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Extracting dataset from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="KKtkzf9xscdAPBPBf4bT")
project = rf.workspace("autonomous-umjvo").project("lane-detection-2-qpx6p")
version = project.version(2)
dataset = version.download("yolov11")
                

if __name__ == "__main__":
    model = YOLO('yolo11n-seg.pt')  # Ensure this path is correct

    # Train the model on the custom dataset from RoboFlow
    results = model.train(
    data='E:/Autonomus Veh/yolo_road/Lane-Detection-2-1/data.yaml',
    epochs=50,
    imgsz=416,
    batch=2,
    lr0=1e-4,  # Reduce initial learning rate
    name='roboflow_yolov8_custom',
    val=True,
)

# After training, load the best weights and test on a new image
model = YOLO('E:/Autonomus Veh/yolo_road/runs/segment/roboflow_yolov8_custom2/weights/best.pt')  # Load the trained model

# # Perform inference on a test image
results = model('E:/Autonomus Veh/yolo_road/Lane-Detection-2-1/test/images')  # Replace with the path to your test image

# save the results in form of image in one folder
save_dir = './predicted_images'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the results in the specified directory
for i, result in enumerate(results):
    # Generate the file path for saving each result
    save_path = os.path.join(save_dir, f"result_{i}.jpg")
    result.plot(save=True, filename=save_path)
                
                