import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Naya import
from model_arch import get_model

data = []
labels = []
classes = 43
path = 'data/Train'

print("Loading Images... thoda dherya rakhein...")
for i in range(classes):
    class_path = os.path.join(path, str(i))
    images = os.listdir(class_path)
    for a in images:
        try:
            image = cv2.imread(os.path.join(class_path, a))
            # Preprocessing: BGR to RGB (OpenCV BGR read karta hai)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (32, 32))
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image {a}")

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# --- YE PART NAYA HAI: DATA AUGMENTATION ---
# Isse model ko alag-alag angles aur zoom se seekhne ko milega
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False, # Traffic signs flip nahi karne chahiye
    fill_mode="nearest"
)

model = get_model()

# Epochs thode badha dete hain (20-25) taaki achhe se seekhe
print("Training shuru ho rahi hai with Augmentation...")
history = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=25 
)

model.save('traffic_classifier.h5')
print("Model Saved as traffic_classifier.h5")