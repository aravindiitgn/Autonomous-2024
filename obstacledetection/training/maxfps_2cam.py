import cv2
import os
import time
from datetime import datetime

def capture_and_save_frames(cam_index1, cam_index2):
    try:
        # Open camera connections
        cap1 = cv2.VideoCapture(cam_index1)
        cap2 = cv2.VideoCapture(cam_index2)

        # Create folders for saving images for each camera
        folder_cam1 = 'frames_camera1'
        folder_cam2 = 'frames_camera2'

        for folder in [folder_cam1, folder_cam2]:
            try:
                os.makedirs(folder)
            except FileExistsError:
                pass

        frame_count_cam1 = 0
        frame_count_cam2 = 0
        start_time = time.time()

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not (ret1 and ret2):
                print("Error reading frames from one of the cameras")
                break

            # Save frames for Camera 1
            timestamp_cam1 = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            frame_filename_cam1 = os.path.join(folder_cam1, f'frame_{frame_count_cam1}_{timestamp_cam1}.png')
            cv2.imwrite(frame_filename_cam1, frame1)

            # Save frames for Camera 2
            timestamp_cam2 = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            frame_filename_cam2 = os.path.join(folder_cam2, f'frame_{frame_count_cam2}_{timestamp_cam2}.png')
            cv2.imwrite(frame_filename_cam2, frame2)

            frame_count_cam1 += 1
            frame_count_cam2 += 1

            # Control frame rate to maximum fps
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            print(f"Frames captured at {fps:.2f} fps")

            # Reset start time for next iteration
            start_time = time.time()

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release camera connections
        cap1.release()
        cap2.release()

    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == "__main__":
    # Specify camera indices
    camera_index1 = 0  # Update with your camera index
    camera_index2 = 1  # Update with your second camera index

    capture_and_save_frames(camera_index1, camera_index2)
