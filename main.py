import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as kimage

#from keras.ops import expand_dims
from numpy import argmax, expand_dims
import os
import json


# Loading the dictionary from the JSON file
current_working_directory = os.getcwd()
model_directory = os.path.join(current_working_directory, 'Models')
# load trained model classes
bird_classes_file_name = os.path.join(model_directory, "bird_classes.json")
with open(bird_classes_file_name, 'r') as json_file:
    bird_classes = json.load(json_file)
bird_species = {int(k): v for k, v in bird_classes.items()}

# load trained model
model_file_name = os.path.join(model_directory, "Bird_EffiB3_DS7_FT.keras")
MODEL = load_model(model_file_name)

def classify_image(image):
    IMG_SIZE = 300
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Preprocess the image
    img_array = kimage.img_to_array(image)
    img_array = expand_dims(img_array, 0)

    predictions = MODEL.predict(img_array)
    spec_pred = argmax(predictions, axis=1)

    prob_list = get_probabilities(predictions[0], 0.1)

    return spec_pred, prob_list

# get probabilities of the best predictions
def get_probabilities(prob_list, trashold):
    #prob_list = model_labels[ind]
    prob_dict = {}
    i = 0
    for prob in prob_list:
        prob_dict[round(100*prob)] = str(bird_species[i]).replace('_', ' ')
        i += 1
    # myKeys = list(prob_dict.keys())
    myKeys = list(num for num in prob_dict.keys() if num > trashold)
    myKeys.sort(reverse = True)
    # myKeys = myKeys[:num_best]  # [cislo for cislo in cisla if cislo > 30]
    sorted_list = [str(i)+'% '+ prob_dict[i] for i in myKeys]
    return sorted_list

def main():
    st.title("Bird Image Classification")

    uploaded_file = st.file_uploader("Vyber obrázek", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        spec_id, predictions = classify_image(image)

        st.subheader("Název druhu:")

        st.write(str(bird_species[int(spec_id)]).replace('_', ' '))

        st.subheader("Přesnost určení:")
        for i, text in enumerate(predictions):
            st.write(f"{i + 1}. {text}")

if __name__ == "__main__":
    main()