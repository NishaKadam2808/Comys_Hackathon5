# Gender Classification Using CNN (MobileNetV2)

This project uses a deep learning model built with **TensorFlow** and **MobileNetV2** to classify human gender (Male or Female) from facial images. It supports both webcam-based real-time predictions and evaluation on test datasets.


# Project Structure

```text
Task_A/
├── gender_classifier.py              # Model training and fine-tuning
├── test_gender_camera.py             # Predict gender using webcam
├── best_gender_model.h5              # Trained model weights
├── README.md                         # Project documentation
├── train/                            # Training images
│   ├── Male/
│   └── Female/
├── val/                              # Validation images
│   ├── Male/
│   └── Female/
├── venv/                             # Python virtual environment
├── model_architecture/               # Image describing model architecture
├── Result/                           # Screenshots and output metrics



#  Requirements

Install the necessary dependencies using:
pip install tensorflow opencv-python scikit-learn numpy



# Model Architecture

Base: MobileNetV2 pretrained on ImageNet

Custom top layers:

- GlobalAveragePooling
- Dense(128) + ReLU + Dropout(0.3)
- Output: Dense(1) + Sigmoid

Note : The output layer is a single neuron with sigmoid activation, returning a probability between 0 and 1 (where **> 0.5 = Female**, **< 0.5 = Male**).



# Training and evaluating the Model :
To train the model execute following commands :
1. .\venv\Scripts\Activate.ps1 

2. python gender_classifier.py

This will:

    Train the model with class weighting and fine-tuning
    Evaluate on validation data
    Save the model to best_gender_model.h5
    Generate classification report and confusion matrix


# Predicting from Webcam
Run this script to use your webcam for live prediction execute following commands :

1. .\venv\Scripts\Activate.ps1
2. python test_gender_camera.py
3. Press 'c' to capture an image
4. Press 'q' to quit

It detects the face, predicts gender, and shows the result with confidence.



# Sample Output

Predicted Gender: Female (96.73% confidence)



# Authors

- **Nisha Kadam**  
  Email: kadamnisha663@gmail.com

- **Kanchan Garad**  
  Email: garadkanchan05@gmail.com

- **Shraddha Nikam**  
  Email: shraddhanikam2005@gmail.com
