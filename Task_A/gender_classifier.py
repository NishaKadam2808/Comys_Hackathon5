import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Paths
#train_dir = 'train'
#val_dir = 'val'
train_dir = 'D:/face/Comys_Hackathon5/Task_A/train'

val_dir = 'D:/face/Comys_Hackathon5/Task_A/val'



image_size = 224
batch_size = 32
epochs = 5

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

for layer in base_model.layers:
    layer.trainable = False

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_gender_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

# Save training history
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

# Final training metrics
train_acc = history.history['accuracy'][-1] * 100
val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n Final Training Accuracy: {train_acc:.2f}%")
print(f" Final Validation Accuracy: {val_acc:.2f}%")

# ======== Calculate Additional Metrics on Validation Set ========
print("\nEvaluating on Validation Set...")

# Get predictions and ground truth
val_generator.reset()
pred_probs = model.predict(val_generator, verbose=1)
y_pred = (pred_probs > 0.5).astype(int).flatten()
y_true = val_generator.classes

# Classification report
report = classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys(), digits=4)
print("\n Classification Report:\n")
print(report)

# Save report to file
with open("validation_classification_report.txt", "w") as f:
    f.write(report)

# Optional: show confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\n Model training complete. Metrics and report saved. Best model: 'best_gender_model.h5'")
