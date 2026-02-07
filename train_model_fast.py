import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

tf.config.optimizer.set_jit(True)

IMG_SIZE = 128
FRAMES_PER_VIDEO = 10
MAX_VIDEOS_PER_CLASS = 150
EPOCHS = 6
BATCH_SIZE = 64

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

        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)

        frame = frame / 255.0
        frames.append(frame)
        i += step

    cap.release()
    return frames

X, y = [], []

for label, cls in enumerate(["real", "fake"]):
    folder = os.path.join("train", cls)
    files = os.listdir(folder)[:MAX_VIDEOS_PER_CLASS]

    for file in tqdm(files, desc=f"Loading {cls}"):
        frames = extract_frames(os.path.join(folder, file))
        if len(frames) == FRAMES_PER_VIDEO:
            X.extend(frames)
            y.extend([label] * FRAMES_PER_VIDEO)

X = np.array(X)
y = np.array(y)

idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save("deepfake_fast.h5")
