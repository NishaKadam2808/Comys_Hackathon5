import os
import cv2
import csv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


model = load_model('face_recognition_model.h5')  


train_dir = 'train'
distorted_dir = 'distorted_output'
output_csv = 'submission_results.csv'


valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')


threshold = 0.5


def get_embedding(image):
    try:
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        embedding = model.predict(image, verbose=0)
        return embedding.flatten()
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None


print("Loading Reference Embeddings...")
identity_embeddings = {}
for identity_folder in os.listdir(train_dir):
    identity_path = os.path.join(train_dir, identity_folder)
    if not os.path.isdir(identity_path):
        continue

    embeddings = []
    for file in os.listdir(identity_path):
        if file.lower().endswith(valid_extensions):
            img_path = os.path.join(identity_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Could not read image: {img_path}")
                continue
            emb = get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
    identity_embeddings[identity_folder] = embeddings

#storing result in submission_results.csv
print("Matching distorted images and writing to CSV...")
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['distorted_image', 'matched_identity', 'label', 'similarity_score'])

    for root, _, files in os.walk(distorted_dir):
        for file_name in files:
            if not file_name.lower().endswith(valid_extensions):
                continue

            distorted_path = os.path.join(root, file_name)
            img = cv2.imread(distorted_path)
            if img is None:
                print(f"[ERROR] Could not read distorted image: {distorted_path}")
                continue

            distorted_embedding = get_embedding(img)
            if distorted_embedding is None:
                continue

            best_score = float('inf')
            best_match = None

            for identity, ref_embeddings in identity_embeddings.items():
                for ref_emb in ref_embeddings:
                    score = np.linalg.norm(distorted_embedding - ref_emb)
                    if score < best_score:
                        best_score = score
                        best_match = identity

            label = 1 if best_score < threshold else 0
            writer.writerow([file_name, best_match, label, f"{best_score:.4f}"])
            print(f"[{label}] Match: {file_name} → {best_match} (Score: {best_score:.4f})")

print(f"\n Completed! Results saved to: {output_csv}")
