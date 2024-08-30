import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
import tkinter.font as font


print(dir(cv2.face))


# Path to Haar cascade file and dataset
CASCADE_PATH = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\haarcascade_frontalface_default.xml'
MODEL_PATH = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\model.yml'
DATASET_PATH = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\persons\lfw-deepfunneled\lfw-deepfunneled'

def train():
    print("Training process started...")

    recog = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset directory '{DATASET_PATH}' not found.")
        return

    faces = []
    ids = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for filename in files:
            try:
                path = os.path.join(root, filename)
                print(f"Processing file: {filename}")

                label = os.path.basename(root)
                parts = filename.split('_')
                if len(parts) < 2:
                    print(f"Filename '{filename}' does not match expected format.")
                    continue

                id_str = parts[-1].split('.')[0]
                try:
                    id = int(id_str)
                except ValueError:
                    print(f"Invalid ID '{id_str}' in filename '{filename}'.")
                    continue

                img = cv2.imread(path, 0)
                if img is None:
                    print(f"Failed to read image '{filename}'.")
                    continue

                faces.append(img)
                ids.append(id)

            except Exception as e:
                print(f"Error processing file '{path}': {e}")

    if not faces or not ids:
        print("No valid images found for training.")
        return

    print(f"Training with {len(faces)} faces.")
    recog.train(faces, np.array(ids))

    try:
        recog.save(MODEL_PATH)
        print(f"Model successfully saved at '{MODEL_PATH}'")
    except Exception as e:
        print(f"Error saving model: {e}")


def identify():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    labelslist = {}
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset directory '{DATASET_PATH}' not found.")
        return

    paths = [os.path.join(root, filename) for root, _, files in os.walk(DATASET_PATH) for filename in files]
    for path in paths:
        try:
            filename = os.path.basename(path)
            parts = filename.split('_')
            if len(parts) < 2:
                print(f"Filename '{filename}' does not match expected format.")
                continue

            id_str = parts[-1].split('.')[0]
            label = parts[0]
            if len(parts) > 2:
                label = '_'.join(parts[:-1])

            labelslist[id_str] = label
        except Exception as e:
            print(f"Error processing file '{path}': {e}")

    print(labelslist)
    recog = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.isfile(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found.")
        return

    print(f"MODEL_PATH: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        print("Model file found.")
    else:
        print("Model file not found.")

    recog.read(MODEL_PATH)

    while True:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            label, confidence = recog.predict(roi)

            if confidence < 100:
                cv2.putText(frm, f"{labelslist.get(str(label), 'Unknown')} {int(confidence)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frm, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Identify", frm)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

def collect_data():
    # Placeholder function, implement data collection logic here
    pass

def maincall():
    root = tk.Tk()
    root.geometry("480x100")
    root.title("Identify")

    label = tk.Label(root, text="Select below buttons ")
    label.grid(row=0, columnspan=2)
    label_font = font.Font(size=35, weight='bold', family='Helvetica')
    label['font'] = label_font

    btn_font = font.Font(size=25)

    button1 = tk.Button(root, text="Add Member", command=collect_data, height=2, width=20)
    button1.grid(row=1, column=0, pady=(10, 10), padx=(5, 5))
    button1['font'] = btn_font

    button2 = tk.Button(root, text="Start with known", command=identify, height=2, width=20)
    button2.grid(row=1, column=1, pady=(10, 10), padx=(5, 5))
    button2['font'] = btn_font

    root.mainloop()

if __name__ == "__main__":
    maincall()

