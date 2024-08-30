import cv2
from datetime import datetime
import os

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def in_out():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Ensure the directories exist
    in_dir = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\visitors\in'
    out_dir = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\visitors\out'
    ensure_directory_exists(in_dir)
    ensure_directory_exists(out_dir)

    right, left = False, False

    while True:
        ret1, frame1 = cap.read()
        if not ret1:
            print("Error: Could not read frame.")
            break
        
        frame1 = cv2.flip(frame1, 1)
        ret2, frame2 = cap.read()
        if not ret2:
            print("Error: Could not read frame.")
            break
        
        frame2 = cv2.flip(frame2, 1)

        diff = cv2.absdiff(frame2, frame1)
        diff = cv2.blur(diff, (5,5))
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshd = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contr, _ = cv2.findContours(threshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x = 300
        if len(contr) > 0:
            max_cnt = max(contr, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Determine direction and save image
        if not right and not left:
            if x > 500:
                right = True
                print("Detected motion to the right.")
            elif x < 200:
                left = True
                print("Detected motion to the left.")
        elif right:
            if x < 200:
                print("Motion detected to left. Saving image...")
                right, left = False, False
                filename = os.path.join(in_dir, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                if cv2.imwrite(filename, frame1):
                    print(f"Image saved successfully: {filename}")
                else:
                    print(f"Error saving image to: {filename}")
        elif left:
            if x > 500:
                print("Motion detected to right. Saving image...")
                right, left = False, False
                filename = os.path.join(out_dir, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                if cv2.imwrite(filename, frame1):
                    print(f"Image saved successfully: {filename}")
                else:
                    print(f"Error saving image to: {filename}")

        # Display the frame
        cv2.imshow("Motion Detection", frame1)
        
        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

