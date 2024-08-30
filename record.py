import cv2
import os
from datetime import datetime

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return None
    return cap

def record():
    # Ensure the recordings directory exists
    recordings_path = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\recordings'
    ensure_directory_exists(recordings_path)

    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        return

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(recordings_path, f'{datetime.now().strftime("%H-%M-%S")}.avi')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    if not out.isOpened():
        print("Error: Video writer could not be opened.")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Add timestamp to frame
        timestamp = datetime.now().strftime("%D-%H-%M-%S")
        cv2.putText(frame, timestamp, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

        # Write frame to video
        out.write(frame)
        
        # Display the frame
        cv2.imshow("Press 'Esc' to stop", frame)

        # Exit on 'Esc' key
        if cv2.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
