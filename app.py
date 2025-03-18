import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from PIL import Image


interpreter = tf.lite.Interpreter(model_path="Brain_T_Model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

def preprocess_image(image):
    """ Preprocess the uploaded image to match TFLite model input requirements. """
    image = np.array(image)
    image = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0).astype(np.float32)  
    return image


st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    processed_image = preprocess_image(image)

    
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(predictions, axis=1)[0]

   
    st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
