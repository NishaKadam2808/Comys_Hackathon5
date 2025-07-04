import os
import cv2
import numpy as np


train_dir = 'train'


valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')


output_dir = 'distorted_output'
os.makedirs(output_dir, exist_ok=True)


def distort_image(image):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, 10], [0, 1, 0]])  
    return cv2.warpAffine(image, M, (cols, rows))


distorted_count = 0


for root, dirs, files in os.walk(train_dir):
    for file in files:
        
        if not file.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(root, file)

        if not os.path.isfile(img_path):
            continue 

        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Could not read image: {img_path}")
            continue

       
        distorted_img = distort_image(image)

        
        relative_path = os.path.relpath(root, train_dir)
        save_dir = os.path.join(output_dir, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        
        save_path = os.path.join(save_dir, f'distorted_{file}')
        cv2.imwrite(save_path, distorted_img)

        distorted_count += 1

print(f"Created {distorted_count} simulated distorted images.")
