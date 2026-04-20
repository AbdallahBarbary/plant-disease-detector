"""
train.py — Plant Disease Detection CNN
=======================================
Uses Transfer Learning with MobileNetV2 pretrained on ImageNet.

SETUP:
1. Download dataset from Kaggle:
   https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Extract so your folder looks like:
   data/
     train/
       Apple___Apple_scab/
       Apple___Black_rot/
       ... (38 classes total)
     valid/
       Apple___Apple_scab/
       ...
3. Run: python train.py

OUTPUT:
   model/plant_disease_model.h5   ← used by app.py
   model/class_names.json         ← class index mapping
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── CONFIG ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 10   # train only the new head first
EPOCHS_FINE = 10   # then fine-tune top layers of MobileNetV2
DATA_DIR    = "data"
MODEL_PATH  = "model/plant_disease_model.h5"
NAMES_PATH  = "model/class_names.json"

# ── DATA AUGMENTATION ──────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "valid"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\n✅ Found {NUM_CLASSES} classes, {train_gen.samples} training images\n")

# Save class names for use in app.py
class_names = {v: k for k, v in train_gen.class_indices.items()}
os.makedirs("model", exist_ok=True)
with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f)
print(f"✅ Class names saved to {NAMES_PATH}")

# ── BUILD MODEL ────────────────────────────────────────────────────────────
# Load MobileNetV2 pretrained on ImageNet, exclude top classification layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze base — only train new head first

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()

# ── PHASE 1: Train head only ───────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
]

print("\n── Phase 1: Training classification head ──\n")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_phase1,
)

# ── PHASE 2: Fine-tune top layers of MobileNetV2 ──────────────────────────
# Unfreeze the top 30 layers of base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # very low LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
]

print("\n── Phase 2: Fine-tuning top MobileNetV2 layers ──\n")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_phase2,
)

# ── EVALUATE ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy: {acc * 100:.2f}%")
print(f"✅ Model saved to: {MODEL_PATH}")
