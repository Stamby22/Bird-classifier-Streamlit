import streamlit as st
from PIL import Image
import keras # tensorflow as tf
from keras.preprocessing import image as kimage
from keras.applications.efficientnet_v2 import decode_predictions
import os
import json

IMG_SIZE = 300
# Loading the dictionary from the JSON file
current_working_directory = os.getcwd()
model_directory = os.path.join(current_working_directory, 'Models')
# load trained model classes
bird_classes_file_name = os.path.join(model_directory, "bird_classes.json")
with open(bird_classes_file_name, 'r') as json_file:
    bird_classes = json.load(json_file)

bird_species = {v: k for k, v in bird_classes.items()}

# load trained model
model_file_name = os.path.join(model_directory, "Bird_EffiB3_DS7_FT.keras")
MODEL = keras.models.load_model(model_file_name)

def classify_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, IMG_SIZE)

    # Preprocess the image
    img_array = kimage.img_to_array(image)
    img_array = keras.expand_dims(img_array, 0)

    predictions = MODEL.predict(img_array)
  
    decoded_predictions = decode_predictions(predictions)[0]
    return decoded_predictions
  
def main():
    st.title("Bird Image Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predictions = classify_image(image)

        st.subheader("Classification Results:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i + 1}: {label} ({score:.2f})")

if __name__ == "__main__":
    main()