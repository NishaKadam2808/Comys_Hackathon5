import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

def evaluate_model(model, data_gen, class_mode='binary'):
    predictions = model.predict(data_gen, verbose=0)
    if class_mode == 'binary':
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(predictions, axis=1)

    y_true = data_gen.classes
    target_names = list(data_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return report

def run_task_a(task_a_path):
    print("Running Task A: Gender Classification")
    model_path = os.path.join("Task_A", "best_gender_model.h5")
    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        task_a_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    report = evaluate_model(model, generator, class_mode='binary')
    print("\nTask A Classification Report:\n")
    print(report)

    with open("taskA_report.txt", "w") as f:
        f.write(report)

def run_task_b(task_b_path):
    print(" Running Task B: Face Recognition")
    model_path = os.path.join("Task_B", "face_recognition_model.h5")
    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        task_b_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    report = evaluate_model(model, generator, class_mode='categorical')
    print("\n Task B Classification Report:\n")
    print(report)

    with open("taskB_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Task A and Task B models on test data")
    parser.add_argument("--task_a_path", required=True, help="Path to Task A test folder")
    parser.add_argument("--task_b_path", required=True, help="Path to Task B test folder")
    args = parser.parse_args()

    run_task_a(args.task_a_path)
    run_task_b(args.task_b_path)
