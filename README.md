
# Autonomous-2024
This project is a result of a full semester course **ME 691-XIII Autonomous Vehicles** offered at IIT Gandhinagar by prof Harish PM. The project is completely student run and still under progress, there are further developments on the way. We have acheived Level 3 Autonomy on an Electric Golf cart with very less hardware changes.

There are multiple sections of the cart on which modifications were done. ![14-seater-electric-golf-cart](https://github.com/user-attachments/assets/39bc530b-ad84-4064-8a0a-6ae69f878818)

## Perception

### Hardware
One Intel RealSense d435 RGBD camera and one Lenovo FHD webcam.

### Software

## Steering

### Hardware
The steering wheel of the cart is attached to a 12V high torque Planetary motor that is controlled via an Arduino and a Cytron 30A motor Driver(Owing to high current draw).![image](https://github.com/user-attachments/assets/823e722e-f74f-49ef-a425-b8ad93724349)
![image](https://github.com/user-attachments/assets/0c9bbbc6-5f9f-4864-8645-1f8a31175819). The Motor Driver is powered through a Genx 10k 25C Lipo Battery.


## Braking

## Localization
### Hardware
Used an Arduino UNO R3 and a Neo 6mv2 GPS Module. The accuracy of this GPS module is close to 2.5~3.5 m while the GPS is fixed to atleast 5 satellites for trilateration.![image](https://github.com/user-attachments/assets/b914471f-1ac3-40b9-9c89-c2872a840783)

### Software
So our aim is to run the cart autonomously anywhere inside our campus. For that we need to know where is it located.
One of the difficulties was to get a map of the campus roads but thanks to OpenStreetMaps - ( an open source maps library) that made our jobs much easier. OpenStreetMaps comes with **osmnx** - a library that u can install directly via pip to your python and start using their functionalities like projection and map inference. ![image](https://github.com/user-attachments/assets/0b734ccb-e23b-4299-ae07-0e699680841a)


First we got the map of interest from their Website( Essentially an xml file that contains maps and its lat,long projections). Simply download the map and next we used a library called **networkx** to convert that map to a Graph of nodes and edges. That helps us in applying any path finding algorithm easily like djikstra or A*. ![image](https://github.com/user-attachments/assets/bddeb112-dd67-4628-8732-1265e47f4673)


Next we have a map of the campus roads visible through a tkinter gui window on which we select the Start and end node and a* algorithm returns a path(Red) that the cart can traverse. Meanwhile we can also show our current location on the map(Blue) by projecting the latitude, longitude values coming from the GPS Module back to the map by **pyproj** library. 

![WhatsApp Image 2024-11-15 at 17 50 44(1)](https://github.com/user-attachments/assets/8e77a232-152f-4668-973d-db460994d7e4)



### Interfacing the GPS Module

Although the gps module works okayish outside, it fails to work accurately when there is something obstructing its view (eg. Trees, Buildings). For most part the accuracy is close to 3 meters.
The GPS module prints the info in NMEA format that needs to be parsed. 

**$GPGGA**,110617.00,41XX.XXXXX,N,00831.54761,W,1,05,2.68,129.0,M,50.1,M,,*42

**110617** – represents the time at which the fix location was taken, 11:06:17 UTC

**41XX.XXXXX,N** – latitude 41 deg XX.XXXXX’ N

**00831.54761,W** – Longitude 008 deg 31.54761′ W

**2.68**– Horizontal dilution of position

**129.0, M** – Altitude, in meters above the sea level

Thankfully there are libraries that parse the data for u (**pynmea2** in python and **TinyGPS++** in Arduino Library Manager)

The cart then follows the path and starts and stops at the respective node point.

## Acceleration

The cart accelerates on a 5v logic circuit. The speed is changed by pressing the throttle potentiometer. 
We bypassed the throttle cables and connected them to an arduino to send pwm to control speed digitally. Higher PWM = Higher speed.
We also implemented a trapizoidal velocity profiling method to increase & decrease the velocity of the cart when an obstacle is detected.



## Power Electronics
![image](https://github.com/user-attachments/assets/5f02721d-c644-43b5-9c12-58185abe1ccf)


## Compute
RTX 4060 

Here’s a comprehensive README file for your code that you can upload to GitHub:

---

# YOLO11n-seg Lane Detection with Custom Dataset

## Features
- Download and use a custom dataset from RoboFlow.
- Train a YOLO11n-seg segmentation model with specified parameters.
- Perform inference on test images using the trained model.
- Save the predicted images to a specified directory.

---

## Prerequisites

1. Python 3.8 or above.
2. Installed libraries:
   - `ultralytics`
   - `cv2` (OpenCV)
   - `matplotlib`
   - `os`
   - `roboflow`
3. GPU support is recommended for training.

---

## Dataset

The dataset is automatically downloaded from RoboFlow using the provided API key and project details.

1. Replace `KKtkzf9xscdAPBPBf4bT` with your RoboFlow API key if needed.
2. The dataset is downloaded and prepared for YOLO training format.

---

## Training the Model

The model is trained using the following parameters:
- **Epochs:** 50
- **Image size:** 416x416
- **Batch size:** 2
- **Initial learning rate:** 0.0001

Modify the training parameters as needed:
```python
results = model.train(
    data='E:/Autonomus Veh/yolo_road/Lane-Detection-2-1/data.yaml',
    epochs=50,
    imgsz=416,
    batch=2,
    lr0=1e-4,
    name='roboflow_yolo11_custom',
    val=True,
)
```

---

## Directory Structure

```
project/
│
├── yolo11n-seg.pt               # YOLO11n model weights
├── dataset/                     # Custom dataset downloaded from RoboFlow
│   ├── train/                   # Training data
│   ├── val/                     # Validation data
│   └── test/                    # Test data
├── predicted_images/            # Directory for storing inference results
└── main.py                      # Script for training and testing the model
```

---


