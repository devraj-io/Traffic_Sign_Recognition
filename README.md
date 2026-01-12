
---

# 🚦 Real-Time Traffic Sign Recognition System

This project is an end-to-end **Deep Learning** solution designed to identify and classify traffic signs into 43 distinct categories using **Convolutional Neural Networks (CNN)**. It features a real-time detection system via webcam and a user-friendly web interface for image uploads.

## ✨ Key Features

* **High Accuracy:** Achieved **98%+ accuracy** using a deep CNN architecture with Batch Normalization.
* **Real-Time Detection:** Integrated with **OpenCV** for live traffic sign recognition from a video stream.
* **Robust Preprocessing:** Utilizes **Data Augmentation** (rotation, zoom, and shifts) to ensure the model performs well under various lighting and angles.
* **Web Interface:** A sleek **Streamlit** dashboard for users to browse and test images from their local machine.

## 🛠️ Technology Stack

* **Programming Language:** Python 3.11
* **Deep Learning Framework:** TensorFlow, Keras
* **Computer Vision:** OpenCV (cv2)
* **Web Framework:** Streamlit
* **Data Analysis:** NumPy, Pandas, Scikit-learn

## 📁 Project Structure

```text
Traffic_Sign_Project/
├── data/               # GTSRB Dataset (folders 0-42)
├── traffic_classifier.h5 # Final trained model weights
├── train.py            # Script for training with Data Augmentation
├── model_arch.py       # CNN architecture definition
├── live_detect.py      # Script for real-time webcam inference
├── app.py              # Streamlit web application script
└── requirements.txt    # Project dependencies

```

## Dataset Link
```
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download
```

## 🚀 Installation & Usage

### 1. Clone the Repository and Install Dependencies

```bash
pip install tensorflow opencv-python streamlit numpy scikit-learn pillow
(you can also follow requirements.txt)
optional: "pip install -r requirements.txt"
```

### 2. Training the Model

To retrain the model with the augmented dataset:

```bash
python train.py

```

### 3. Launching the Web App

To use the browser-based "Browse and Predict" tool:

```bash
streamlit run app.py

```

### 4. Running Real-Time Detection

To detect signs via your computer's webcam:

```bash
python live_detect.py

```

## 🧠 Technical Overview

* **Architecture:** The model consists of multiple Convolutional layers for feature extraction, followed by **Batch Normalization** for training stability and **Dropout** layers to prevent overfitting.
* **Dataset:** Trained on the **GTSRB (German Traffic Sign Recognition Benchmark)**, containing over 50,000 images.
* **Optimization:** Uses the **Adam optimizer** and **Categorical Crossentropy** loss function for multi-class classification.

## 🎯 Results

The system successfully distinguishes between similar-looking signs (e.g., different speed limits) and performs reliably on high-resolution images as well as live camera feeds, even in the presence of glare or minor obstructions.

---
