import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



test_dir = 'D:/face/Comys_Hackathon5/Task_A/train'  


img_size = (224, 224)
batch_size = 32


model = load_model('best_gender_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  
)


predictions = model.predict(test_data)
y_pred = (predictions > 0.5).astype(int).reshape(-1)
y_true = test_data.classes


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n--- Evaluation Metrics on Test Set ---")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_true, y_pred, target_names=['Female', 'Male']))
