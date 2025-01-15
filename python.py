import os 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical 

DATA_DIR = 'Dataset'
CATEGORIES = ['cataract', 'normal']
IMG_SIZE = 224

def load_data():
    data = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img) / 255.0
                data.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    data = np.array(data)
    labels = to_categorical(labels, num_classes=2)
    return train_test_split(data, labels, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()
