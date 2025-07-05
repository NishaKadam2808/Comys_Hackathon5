
# face recognition and verification with distorted inputs

# Overview
This project builds a face recognition system capable of verifying identities, even from distorted or altered face images.

# Project Structure
```text
Task_B/
├── train/                         # Folder with clean reference identity images
├── distorted_output/              # Will be created for storing distorted images
├── face_recognition_model.h5      # Pretrained model file (will be created after training)
├── generate_distorted_images.py   # Script to create distorted images
├── training_script.py             # Script to train model on distorted data
├── Inference_Pipeline_Code_for_Face_Verification.py  # Face matching script
├── README.md
|__ venv/
|__ submission_results
```

# Task Objective
- Identify whether a distorted image matches any identity in the `train` folder.
- Outputs a CSV file with predicted identity and match score.

# Technologies Requirements 
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn


# How to Run :

Step 1 : Before executing following commands execute :  .\venv\Scripts\Activate.ps1 

Step 2 : Generate Distorted Images
         python Inference_Pipeline_Code_for_Face_Verification.py

Step 3:  Train and evaluate the Face Recognition Model
         python training_script.py
         # The model will be saved as face_recognition_model.h5



Step 4:  Run Face Verification on Distorted Images
         python Inference_Pipeline_Code_for_Face_Verification.py

         This will create a submission_results.csv containing:

            -Distorted image name

            -Matched identity

            -Label (1 = match, 0 = no match)

            -Similarity score


# Output
 `submission_results.csv`: Contains predicted labels for distorted images





# Authors

- **Nisha Kadam**  
  Email: kadamnisha663@gmail.com

- **Kanchan Garad**  
  Email: garadkanchan05@gmail.com

- **Shraddha Nikam**  
  Email: shraddhanikam2005@gmail.com
