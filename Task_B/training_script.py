import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


dataset_dir = "distorted_output"
image_size = 224
batch_size = 32
num_epochs = 20


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)


for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("face_recognition_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs,
    callbacks=[checkpoint]
)

print("Model training complete and best version saved as 'face_recognition_model.h5'.")
