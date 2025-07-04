import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



train_dir = 'D:/face/Comys_Hackathon5/Task_A/train'

val_dir = 'D:/face/Comys_Hackathon5/Task_A/val'



img_size = (224, 224)
batch_size = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_gender_model.h5', save_best_only=True)
]


print("Training frozen MobileNetV2...")
model.fit(train_data, validation_data=val_data, epochs=20, callbacks=callbacks)


val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy after initial training: {val_acc * 100:.2f}%")


print("Fine-tuning MobileNetV2...")
base_model.trainable = True


model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)


val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy after fine-tuning: {val_acc * 100:.2f}%")
