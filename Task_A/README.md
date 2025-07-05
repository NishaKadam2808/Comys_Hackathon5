# Gender Classification Using CNN (MobileNetV2)

This project uses a deep learning model built with **TensorFlow** and **MobileNetV2** to classify human gender (Male or Female) from facial images. It supports both webcam-based real-time predictions and evaluation on test datasets.


# Project Structure

Task_A/
├── gender_classifier.py # Model training, fine-tuning, evaluation
├── test_gender_camera.py # Webcam-based gender prediction
├── best_gender_model.h5 # Saved trained model
├── README.md # Project documentation
├── train/ # Training images
│ ├── Male/
│ └── Female/
├── val/ # Validation images
│ ├── Male/
│ └── Female/
├── venv/ # Python virtual environment
├── model_architecture/ # Diagram showing model structure
├── Result/ # Screenshots and evaluation output 

#  Requirements

Install the necessary dependencies using:
pip install tensorflow opencv-python scikit-learn numpy



# Model Architecture
Base: MobileNetV2 pretrained on ImageNet

Custom top layers:

GlobalAveragePooling

Dense(128) + ReLU + Dropout(0.3)

Output: Dense(1) + Sigmoid

# Training and Evaluating the Model :
To train the model execute following commands :
1. .\venv\Scripts\Activate.ps1 

2. python gender_classifier.py

This will:

   - Train the model with class weighting and fine-tuning
   - Evaluate on validation data
   - Save the model to best_gender_model.h5
   - Generate classification report and confusion matrix









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