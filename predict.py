import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('traffic_classifier.h5')

def predict_sign(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0) / 255.0
    
    # Prediction logic
    prediction = model.predict(img)
    class_id = np.argmax(prediction)

# Yahan ID ki jagah name nikaal rahe hain
    sign_name = classes[class_id] 

    print(f"Sign Detected: {sign_name}")
# Agar Streamlit hai toh: st.success(f"Result: {sign_name}")

# Ise use karne ke liye:
# predict_sign('path_to_your_test_image.png')