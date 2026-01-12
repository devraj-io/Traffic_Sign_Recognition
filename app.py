import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Model load karo
model = tf.keras.models.load_model('traffic_classifier.h5')

# Saari 43 Classes ke names
classes = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
    41:'End of no passing', 42:'End no passing veh > 3.5 tons' 
}

# Web App ka Title
st.set_page_config(page_title="Traffic Sign Detector", layout="centered")
st.title("🚦 Traffic Sign Recognition")
st.markdown("Computer se Traffic Sign ki image **Browse** karein aur dekhein model usey pehchaanta hai ya nahi.")

# File Uploader (Browse Button)
uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Image ko show karna
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    # 2. Preprocessing
    img = image.resize((32, 32))
    img = np.array(img)
    # Check if image is RGB, if not convert it (extra safety)
    if img.shape[-1] != 3:
        img = img[:,:,:3]
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 3. Predict Button
    if st.button('Predict Sign'):
        prediction = model.predict(img)
        class_id = np.argmax(prediction)
        sign_name = classes[class_id]
        
        # Result display karna
        st.success(f"**Prediction: {sign_name}**")
        st.info(f"Class ID: {class_id}")