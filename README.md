# COMYS Hackathon 5 – AI Model Submission

This repository contains our submission for the COMYS Hackathon 5, comprising:

- **Task A** – Gender Classification using CNN (MobileNetV2)
- **Task B** – Face Recognition and Verification with Distorted Inputs

Both models are fully trained and evaluated. The `test.py` script can be used to reproduce final metrics using the submitted `.h5` model weights.

# Project Structure

```

Comys\_Hackathon5/
├── test.py                            # Main evaluation script (required for execution)
├── README.md                          # Root-level documentation
├── Task\_A/
│   ├── best\_gender\_model.h5           # Trained model for gender classification
│   ├── gender\_classifier.py           # (Training script – not used in evaluation)
│   ├── train/                          # Test images (provided by organizers)
│   └── README.md                      # Task A documentation
├── Task\_B/
│   ├── face\_recognition\_model.h5      # Trained model for face verification
│   ├── training\_script.py             # (Training script – not used in evaluation)
│   ├── train/                          # Test images (provided by organizers)
│   └── README.md                      # Task B documentation
├── taskA\_report.txt                   # Classification report for Task A (auto-generated)
├── taskB\_report.txt                   # Classification report for Task B (auto-generated)

```


# Setup Instructions

Ensure Python 3.9+ and pip are installed

pip install tensorflow scikit-learn numpy opencv-python

# Running the Evaluation

The `test.py` script is the only entry point used by organizers. It evaluates both Task A and Task B using your submitted models and test folders.

python test.py --task_a_path Task_A/train --task_b_path Task_B/train

This will:

* Load the models from the `.h5` files
* Run inference on the test folders (same format as your `val/`)
* Output classification metrics to the console
* Save reports as:

  * `taskA_report.txt`
  * `taskB_report.txt`

# Output Format

Each report contains:

* Accuracy
* Precision
* Recall
* F1-Score (macro and weighted)
* Support count per class

Example extract from `taskA_report.txt`:

```
              precision    recall  f1-score   support

      female     0.8950    0.7792    0.8331       394
        male     0.9450    0.9765    0.9605      1532

    accuracy                         0.9361      1926
   macro avg     0.9200    0.8778    0.8968      1926
weighted avg     0.9348    0.9361    0.9345      1926
```

# Reproducibility Policy

* No re-training occurs in `test.py`
* The models used are:

  * `Task_A/best_gender_model.h5`
  * `Task_B/face_recognition_model.h5`
* The script directly loads these weights and performs inference only

> Results will match those submitted, ensuring full compliance with competition rules.

# Team Members

* **Nisha Kadam**
   [kadamnisha663@gmail.com](mailto:kadamnisha663@gmail.com)

* **Kanchan Garad**
   [garadkanchan05@gmail.com](mailto:garadkanchan05@gmail.com)

* **Shraddha Nikam**
   [shraddhanikam2005@gmail.com](mailto:shraddhanikam2005@gmail.com)


