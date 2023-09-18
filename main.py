import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
import cv2
import os
import numpy as np

train2 = ['Seborrheic Keratoses and other Benign Tumors',
 'Psoriasis pictures Lichen Planus and related diseases',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Eczema Photos',
 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions']

vgg_model = tf.keras.applications.vgg19.VGG19(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3))

def predict_skin_disease(image_path):
   # Define list of class names
    class_names = train2

    # Load saved model
    model = tf.keras.models.load_model('skin.h5')

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    # Make prediction on preprocessed image
    pred = model.predict(img)[0]
    print(pred)
    predicted_class_index = np.argmax(pred)
    print(predicted_class_index)
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

# print(predict_skin_disease(r"D:\Downloads\contest\SIH\Data Set Skin Disease\train\Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions\actinic-cheilitis-sq-cell-lip-1.jpg"))

# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
 
# Declaring our FastAPI instance
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def image_detect(image_path) :
    return {"prediction" : predict_skin_disease(image_path)}
