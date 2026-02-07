import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)

tf.config.optimizer.set_jit(True)

IMG_SIZE = 128
FRAMES_PER_VIDEO = 8
EPOCHS = 7
BATCH_SIZE = 64
VAL_VIDEO_COUNT = 100

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
    jitter = np.random.randint(-2, 3, size=len(indices))
    indices = np.clip(indices + jitter, 0, length - 1)

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

def load_videos(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]

real_videos = load_videos("train/real")
fake_videos = load_videos("train/fake")

real_train, real_val = train_test_split(
    real_videos, test_size=VAL_VIDEO_COUNT // 2, random_state=42
)
fake_train, fake_val = train_test_split(
    fake_videos, test_size=VAL_VIDEO_COUNT // 2, random_state=42
)

train_videos = [(v, 0) for v in real_train] + [(v, 1) for v in fake_train]
val_videos   = [(v, 0) for v in real_val]   + [(v, 1) for v in fake_val]

print(f"Training videos: {len(train_videos)}")
print(f"Validation videos: {len(val_videos)}")

X_train, y_train = [], []

for video, label in tqdm(train_videos, desc="Loading training data"):
    frames = extract_frames(video)
    if len(frames) >= FRAMES_PER_VIDEO // 2:
        X_train.extend(frames)
        y_train.extend([label] * len(frames))

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Total training frames:", len(X_train))
if len(X_train) == 0:
    raise RuntimeError("No training frames extracted")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.35)(x)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(1e-5),
    loss=BinaryCrossentropy(label_smoothing=0.02),
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

correct = 0

for video, label in tqdm(val_videos, desc="Validating"):
    frames = extract_frames(video)
    if len(frames) < 3:
        continue

    preds = model.predict(np.array(frames), verbose=0).flatten()
    video_prob = np.mean(np.sort(preds)[-3:])   # 🔒 stable aggregation
    pred_label = 1 if video_prob >= 0.5 else 0

    if pred_label == label:
        correct += 1

val_accuracy = correct / len(val_videos)
print(f"\n📊 FINAL VIDEO-LEVEL VALIDATION ACCURACY: {val_accuracy * 100:.2f}%")

model.save("deepfake_fast.h5")
