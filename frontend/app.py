import streamlit as st
import numpy as np
import pandas as pd
import base64
import requests
import io
from PIL import Image
from data import all_mushroom_tables, all_info_tables
import os


st.set_page_config(layout="wide", page_title= "FungAI", page_icon = ":shark:")


# This is creating the headlines
st.markdown("""<div class='title'>This is FungAI 🧠<br>
We analyse your mushrooms 🍄<br>
Please upload your picture! </div>""" , unsafe_allow_html = True)

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

    .title {{
        background-color: white;
        color: black;
        font-size: 60px;
        opacity: 0.8;
        font-family: sans serif arvo;
        border-collapse: separate;
        border-spacing: 15px 15px;
        text-align: center;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(10,10,10,0.05))
    }}

    .stButton {{
        font-size: 40px;
        text-align: center;
        font-family: sans serif arvo;
        margin-top: 10px;
        opacity: 0.8;
            }}

    #picture {{
        background-color: white;
        color: black;
        font-size: 20 px;
        opacity: 0.8;
        font-family: sans serif arvo;
        border-collapse: separate;
        border-spacing: 15px 15px;
        text-align: center;
        opacity: 0.8;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(10,10,10,0.05))
    }}

    #aditional_information {{
        font-size:30px;
        text-align: center;
        font-family: sans serif arvo;
        margin-top: 10px;
        opacity: 0.8;
            }}

    #examples {{
        background-color: white;
        color: black;
        font-size:40px;
        font-family: sans serif arvo;
        margin-bottom:20px;
        text-align: center;
        opacity:0.8;
        border-collapse: separate;
        border-spacing: 15px 15px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(10,10,10,0.05))
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('images_for_app/background2.jpg')




# This is creating the picture upload button
uploaded_file = st.file_uploader("Choose a file")
button = False

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    buf = io.BytesIO(bytes_data)
    image = Image.open(buf)

    #st.write(f'{type(image)}')
    col1, col2, col3  = st.columns(3)

    with col2:
        st.image(image, width=250)
        original_title = '<div id="picture">This is your picture</div>'
        st.markdown(original_title, unsafe_allow_html=True)
        button = st.button("analyse")




# Print alayse and additional information

if button and uploaded_file is not None:
    url = 'https://fungai-tec6gbmrsa-ew.a.run.app/predict'
    #url = 'http://127.0.0.1:1234/predict'
    file = {"image" : bytes_data}
    response = requests.post(url, files=file)

    if response.status_code == 200:
        genuses = response.json()["genuses"]

        predicted_genus = max(genuses, key=genuses.get)
        st.write(f'{predicted_genus}')


        with st.expander("see additional information"):
            #html_string = "<p>style=“background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;“</p>"
        # st.markdown(html_string, unsafe_allow_html=True)

            st.markdown(f"""<div id="aditional_information">{all_info_tables.get(predicted_genus)} </div>""" , unsafe_allow_html = True)


        col1, col2, col3 = st.columns(3)
        dir_gens = f'images_for_app/Genus pictures/{predicted_genus}'
        images = os.listdir(dir_gens)

        with col1:
            image_name = images[0].replace('.jpeg', '').replace('_', ' ').title()
            st.markdown(f"""<div id="examples"> {image_name} </div>""" , unsafe_allow_html = True)

            image1 = Image.open(os.path.join(dir_gens, images[0]))
            st.image(image1, use_column_width=True)

        with col2:
            image_name = images[1].replace('.jpeg', '').replace('_', ' ').title()
            st.markdown(f"""<div id="examples"> {image_name} </div>""" , unsafe_allow_html = True)
            image1 = Image.open(os.path.join(dir_gens, images[1]))
            st.image(image1, use_column_width=True)
        with col3:
            image_name = images[2].replace('.jpeg', '').replace('_', ' ').title()
            st.markdown(f"""<div id="examples"> {image_name} </div>""" , unsafe_allow_html = True)
            image1 = Image.open(os.path.join(dir_gens, images[2]))
            st.image(image1, use_column_width=True)

        st.write()


            # Show receipes button
        # recipes_button = st.button("Reveal recipes")
        # choices = st.radio( "Which recipes do u want?" )
        col1, col2 = st.columns(2)
        dir_recs = f'images_for_app/Recipe images/{predicted_genus}'
        images = [img for img in os.listdir(dir_recs) if not img.startswith(".DS")]

        with col1:
                st.markdown(f"""<div id="examples"> Shroooms Everywhere! </div>""" , unsafe_allow_html = True)
                image1 = Image.open(os.path.join(dir_recs, images[0]))
                st.image(image1, use_column_width=True)
                st.markdown(all_mushroom_tables.get(predicted_genus).get(1),unsafe_allow_html=True)

        with col2:
                st.markdown(f"""<div id="examples"> Sweet Oh Recipe! </div>""" , unsafe_allow_html = True)
                image1 = Image.open(os.path.join(dir_recs, images[1]))
                st.image(image1, use_column_width=True)
                st.markdown( all_mushroom_tables.get(predicted_genus).get(2),unsafe_allow_html=True)

    # with col3:
    #         st.header("Recipe 3")
    #         image1 = Image.open(os.path.join(dir_recs, images[3]))
    #         st.image(image1)
    #         st.markdown( all_mushroom_tables.get(predicted_genus).get(3),unsafe_allow_html=True)
    else:
        st.warning("WoOoops! It seems like something went wrong!")
        st.warning("Click again to retry!")
