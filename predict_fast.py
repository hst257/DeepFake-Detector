import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

IMG_SIZE = 128
FRAMES_PER_VIDEO = 10

model = tf.keras.models.load_model("deepfake_fast.h5")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(length // FRAMES_PER_VIDEO, 1)

    i = 0
    while len(frames) < FRAMES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
        i += step

    cap.release()
    return frames

df = pd.read_csv("test_public.csv")

labels = []
probabilities = []

for video in tqdm(df["filename"]):
    frames = extract_frames(os.path.join("test", video))

    if len(frames) < FRAMES_PER_VIDEO:
        labels.append(0)
        probabilities.append(0.0)
        continue

    preds = model.predict(np.array(frames), verbose=0).flatten()

    topk = np.sort(preds)[-3:]
    video_prob = float(np.mean(topk))

    label = 1 if video_prob > 0.4 else 0

    labels.append(label)
    probabilities.append(video_prob)

df["label"] = labels
df["probability"] = probabilities

df.to_csv("submission.csv", index=False)
