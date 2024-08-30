import cv2

# Define the path to the existing model
MODEL_PATH = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\model.yml'

# Define the path where you want to save the new model
NEW_MODEL_PATH = r'C:\Users\nirmi\OneDrive\Desktop\project\smart-cctv-ver2.0\new_model.yml'

# Create the LBPHFaceRecognizer object
recog = cv2.face.LBPHFaceRecognizer_create()

try:
    # Load the existing model
    recog.read(MODEL_PATH)
    print("Model loaded successfully.")

    # Save the model to a new file
    recog.save(NEW_MODEL_PATH)
    print(f"Model saved as '{NEW_MODEL_PATH}'.")
except cv2.error as e:
    print(f"Error loading or saving model: {e}")
