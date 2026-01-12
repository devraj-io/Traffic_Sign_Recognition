import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Model load karo
model = load_model('traffic_classifier.h5')

# 2. Saari 43 Classes ke names (Dictionary)
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

def predict_sign(image_path):
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV BGR read karta hai, CNN RGB pe train hua hai
        img_res = cv2.resize(img_rgb, (32, 32))
        img_final = np.expand_dims(img_res, axis=0) / 255.0
        
        # Prediction logic
        prediction = model.predict(img_final)
        class_id = np.argmax(prediction)

        # Name nikaalna
        sign_name = classes[class_id] 
        print(f"\n[RESULT] Predicted Sign: {sign_name} (Class ID: {class_id})")
        
    except Exception as e:
        print(f"Error: {e}. Check if image path is correct!")

# --- TEST KARNE KE LIYE ---
# Yahan apni image ka sahi path daalo
predict_sign('data/Test/00093.png')
