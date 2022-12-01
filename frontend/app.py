import streamlit as st
import numpy as np
import pandas as pd
import base64
import random
import json
import requests
from google.oauth2 import service_account
from google.cloud import storage
from io import StringIO
import io
from PIL import Image
import matplotlib.pyplot as plt
from data import *
st.set_page_config(layout="wide")
predicted_genus = "Amanita"


# This is creating the headlines
st.markdown("""# This is FungAI
## We analyse your mushrooms
Please upload your picture """)

df = pd.DataFrame({
    'first column': list(range(1, 11)),
    'second column': np.arange(10, 101, 10)
})

# adding background picture
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{encoded_string.decode()});
        background-size: cover
    }}

    [data-testid='stExpander'] {{
        background-color:white;
        }}

    .info {{
        background-color: white;
        color: black;
        border:2px solid gray;
        border-collapse: separate;
        border-spacing: 15px 15px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(10,10,10,0.05))
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('frontend/images_for_app/background2.jpg')


# This is creating the picture upload button
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    buf = io.BytesIO(bytes_data)
    image = Image.open(buf)

    fig , ax  = plt.subplots(nrows = 1 , ncols = 1)
    ax.imshow(image , cmap = "gray")
    st.pyplot(fig)

    # Print additional information button

    # add code


with st.expander("see additional information"):
    #html_string = "<p>style=“background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;“</p>"
   # st.markdown(html_string, unsafe_allow_html=True)
    st.markdown(all_info_tables.get(predicted_genus),
                    unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
        st.header("Amanita Bisporigera")
        image1 = Image.open(f'frontend/images_for_app/genus_pictures/{predicted_genus}/amanita_bisporigera.jpeg')
        st.image(image1, width=250)
with col2:
        st.header("Amanita Caesarea")
        image2 = Image.open(f'frontend/images_for_app/genus_pictures/{predicted_genus}/Amanita_caesarea.jpeg')
        st.image(image2, width=250)
with col3:
        st.header("Amanita Muscaria")
        image3 = Image.open(f'frontend/images_for_app/genus_pictures/{predicted_genus}/amanita_muscaria.jpeg')
        st.image(image3, width=250)

st.write()


    # Show receipes button
# recipes_button = st.button("Reveal recipes")
# choices = st.radio( "Which recipes do u want?" )
col1, col2, col3 = st.columns(3)

with col1:
        st.header("Recipe 1")
        image = Image.open("frontend/images_for_app/recipe1_amanita.png")
        st.image(image)
        st.markdown( all_mushroom_tables.get(predicted_genus).get(1),unsafe_allow_html=True)

with col2:
        st.header("Recipe 2")
        image = Image.open("frontend/images_for_app/recipe2_amanita.png")
        st.image(image)
        st.markdown( all_mushroom_tables.get(predicted_genus).get(2),unsafe_allow_html=True)

with col3:
        st.header("Recipe 3")
        st.markdown( all_mushroom_tables.get(predicted_genus).get(3),unsafe_allow_html=True)




url = 'http://localhost:8000/predict'

params = {
    'new_image': None
}

response = requests.get(url, params=params)
response.json() #=> {wait: 64}
