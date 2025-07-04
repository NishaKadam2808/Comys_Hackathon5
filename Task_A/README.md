# Gender Classification Using CNN (MobileNetV2)

This project uses a deep learning model built with **TensorFlow** and **MobileNetV2** to classify human gender (Male or Female) from facial images. It supports both webcam-based real-time predictions and evaluation on test datasets.


# Project Structure

Task_A/
├── gender_classifier.py # Model training and fine-tuning
├── evaluate_gender_model.py # Evaluation on test data (accuracy, precision, recall, F1)
├── test_gender_camera.py # Predict gender using webcam
├── best_gender_model.h5 # Trained model weights
├── README.md # Project documentation
├── train/
│ ├── Male/
│ └── Female/
├── val/
│ ├── Male/
│ └── Female/
|__venv
|__model_architecture #image describing architecture of our model
|__Result #screenshot of output and accuracy matrics


#  Requirements

Install the necessary dependencies using:
pip install tensorflow opencv-python scikit-learn numpy



# Model Architecture
Base: MobileNetV2 pretrained on ImageNet

Custom top layers:

GlobalAveragePooling

Dense(128) + ReLU + Dropout(0.3)

Output: Dense(1) + Sigmoid

# Training the Model :
To train the model execute following commands :
1. .\venv\Scripts\Activate.ps1 

2. python gender_classifier.py

# The model will be saved as best_gender_model.h5


# Evaluating the Model
To evaluate the trained model on a test dataset :
1. .\venv\Scripts\Activate.ps1 

2. python evaluate_gender_model.py


--- Evaluation Metrics on Test Set ---
Accuracy : 97.62%
Precision: 96.88%
Recall   : 98.24%
F1-Score : 97.55%


# Predicting from Webcam
Run this script to use your webcam for live prediction execute following commands :

1. .\venv\Scripts\Activate.ps1
2. python test_gender_camera.py
3. Press 'c' to capture an image
4. Press 'q' to quit

It detects the face, predicts gender, and shows the result with confidence.



# Sample Output

Predicted Gender: Female (96.73% confidence)

# Author
Nisha
mail: kadamnisha663@gmail.com