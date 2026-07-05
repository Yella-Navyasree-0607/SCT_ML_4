# Task 4 - Hand Gesture Recognition
# SkillCraft Technology Machine Learning Internship

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python.vision import RunningMode
import mediapipe as mp
import urllib.request
import os

# Download hand gesture model
model_path = 'gesture_recognizer.task'
if not os.path.exists(model_path):
    print("Downloading gesture recognizer model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
        model_path
    )
    print("Model downloaded!")

# Setup
gestures = ['None', 'Closed_Fist', 'Open_Palm', 'Thumbs_Up', 'Victory', 'ILoveYou']
BaseOptions = mp.tasks.BaseOptions
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

print("Starting webcam... Press Q to quit")
cap = cv2.VideoCapture(0)

gesture_counts = {g: 0 for g in gestures}

with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        gesture_name = "No hand detected"
        if result.gestures:
            gesture_name = result.gestures[0][0].category_name
            gesture_counts[gesture_name] = gesture_counts.get(gesture_name, 0) + 1

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to quit", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Plot results
detected = {k: v for k, v in gesture_counts.items() if v > 0}
if detected:
    plt.figure(figsize=(8, 5))
    plt.bar(detected.keys(), detected.values(), color='steelblue')
    plt.title('Detected Hand Gestures')
    plt.xlabel('Gesture')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('gesture_results.png')
    plt.show()
    print("Gesture detection complete!")
else:
    print("No gestures detected")