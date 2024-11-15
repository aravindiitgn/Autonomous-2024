import cv2

def display_three_cameras(camera_indexes):
    # Open cameras
    cameras = [cv2.VideoCapture(index) for index in camera_indexes]

    # Set the frame width and height for all cameras
    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit...")

    while True:
        # Capture frames from all cameras
        frames = [camera.read()[1] for camera in cameras]

        # Check if frames are valid before displaying
        for i, frame in enumerate(frames):
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                cv2.imshow(f"Camera {i}", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    for camera in cameras:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define camera indexes
    camera_indexes = [1, 0, 2]  # Update with your camera indexes

    # Display frames from three cameras simultaneously
    display_three_cameras(camera_indexes)
