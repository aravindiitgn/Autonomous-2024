import cv2
import os
import time
from datetime import datetime

def capture_and_save_frames(cam_index):
    try:
        # Open camera connection
        cap = cv2.VideoCapture(cam_index)

        # Create folder for saving images
        folder_name = 'frames'
        try:
            os.makedirs(folder_name)
        except FileExistsError:
            pass

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error reading frames from the camera")
                break

            # Save frame with timestamp and frame number in the filename
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            frame_filename = os.path.join(folder_name, f'frame_{frame_count}_{timestamp}.png')
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

            # Control frame rate to maximum fps
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            print(f"Frame {frame_count} captured at {fps:.2f} fps")

            # Reset start time for next iteration
            start_time = time.time()

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release camera connection
        cap.release()

    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == "__main__":
    # Specify camera index
    camera_index = 0  # Update with your camera index

    capture_and_save_frames(camera_index)
