import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as kimage
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
IMG_SIZE = 300

def classify_image(image):
    image = image.convert("RGB")

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

# Upload an image and set some options for demo purposes
st.header("Určování druhů ptáků ČR")
img_file = st.sidebar.file_uploader(label='Vyber soubor', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Real Time překreslení", value=False)
box_color = st.sidebar.color_picker(label="Barva rámečku", value='#0000FF')


if img_file:
    st.write("Ohraničte objekt zájmu")
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click k uložení výřezu")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=(1, 1))
    
    # Manipulate cropped image at will
    cropped_img = cropped_img.resize((IMG_SIZE, IMG_SIZE))
    st.write("Preview")
    # _ = cropped_img.thumbnail((150,150))
    st.image(cropped_img)

    # Získání souřadnic ohraničeného objektu
    if cropped_img:
        spec_id, predictions = classify_image(cropped_img)

        st.subheader("Název druhu:")
        st.write(str(bird_species[int(spec_id)]).replace('_', ' '))
        
        st.subheader("Přesnost určení:")
        for i, text in enumerate(predictions):
            st.write(f"{i + 1}. {text}")

#if __name__ == "__main__":
#    main()