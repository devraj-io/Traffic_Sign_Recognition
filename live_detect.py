import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Model load karo
model = load_model('traffic_classifier.h5')

# 2. SARE 43 NAMES KI DICTIONARY (Taaki ID na dikhe)
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

# 3. Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img_original = cap.read()
    if not success:
        break
        
    # Preprocessing
    img = np.asarray(img_original)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)
    
    # Prediction
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    probability_value = np.amax(prediction)
    
    # Check if model is confident
    if probability_value > 0.75:
        # Yahan hum dictionary se NAME nikaal rahe hain
        sign_name = classes.get(class_index, "Unknown")
        
        # Screen par text likhna
        cv2.rectangle(img_original, (0, 0), (640, 70), (0, 0, 0), -1) # Background bar for text
        cv2.putText(img_original, f"SIGN: {sign_name}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_original, f"CONFIDENCE: {round(probability_value*100, 2)}%", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Traffic Sign Real-time", img_original)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()