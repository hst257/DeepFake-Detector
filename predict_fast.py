import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tqdm import tqdm

IMG_SIZE = 128
FRAMES_PER_VIDEO = 8
THRESHOLD = 0.5

model = tf.keras.models.load_model("deepfake_fast.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        cap.release()
        return frames

    indices = np.linspace(0, length - 1, FRAMES_PER_VIDEO, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                frame = face

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)

    cap.release()
    return frames

df = pd.read_csv("test_public.csv")

labels = []
probabilities = []

for video in tqdm(df["filename"]):
    frames = extract_frames(os.path.join("test", video))

    if len(frames) < 3:
        labels.append(0)
        probabilities.append(0.0)
        continue

    preds = model.predict(np.array(frames), verbose=0).flatten()
    video_prob = np.mean(np.sort(preds)[-3:])

    labels.append(1 if video_prob >= THRESHOLD else 0)
    probabilities.append(video_prob)

df["label"] = labels
df["probability"] = probabilities
df.to_csv("submission.csv", index=False)
