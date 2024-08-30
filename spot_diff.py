import cv2
import time
from skimage.metrics import structural_similarity
from datetime import datetime
import os
import beepy

# Path to the stolen folder
STOLEN_PATH = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\stolen'

def spot_diff(frame1, frame2):
    if frame1 is None or frame2 is None:
        print("Error: One of the frames is empty. Exiting...")
        return

    frame1 = frame1[1]
    frame2 = frame2[1]

    if frame1 is None or frame2 is None:
        print("Error: One of the frames is empty after splitting. Exiting...")
        return

    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    g1 = cv2.blur(g1, (2,2))
    g2 = cv2.blur(g2, (2,2))

    (score, diff) = structural_similarity(g2, g1, full=True)

    print("Image similarity", score)

    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)[1]

    contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contors = [c for c in contors if cv2.contourArea(c) > 50]

    if len(contors):
        for c in contors:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Generate the filename with the current timestamp
        filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".jpg"
        filepath = os.path.join(STOLEN_PATH, filename)

        # Save the image in the stolen folder
        cv2.imwrite(filepath, frame1)
        print(f"Stolen image saved to {filepath}")

        # Play a beep sound
        beepy.beep(sound=4)
    else:
        print("Nothing stolen")
        return 0

    cv2.imshow("diff", thresh)
    cv2.imshow("win1", frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 1

def find_motion():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    ret1, frm1 = cap.read()
    time.sleep(1)
    ret2, frm2 = cap.read()

    if not ret1 or not ret2:
        print("Error: Could not read frames.")
        cap.release()
        return

    spot_diff((ret1, frm1), (ret2, frm2))
    cap.release()

