import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from home import home_page
from about import about_page
from metrics import metrics_page
from classify import classify_page


# Create two columns
col1, col2 = st.columns([1, 5])

# Image column
with col1:
    img = Image.open("imgs/team16_6_Logo.png")  # Use the relative path or absolute path
    st.image(img, use_container_width=True)

# Title column
with col2:
    st.title('Toxic Comment Classification System')


# Option menu
selected = option_menu(menu_title=None, options=["Home", "About Us", 'Metrics', "Classify"],
                       menu_icon="cast", default_index=0, icons=['house', 'people', 'clipboard-data', 'play'],
                       orientation="horizontal")


# Home page
if selected == 'Home':
    home_page()


# Team page
if selected == 'About Us':
    about_page()


# Metric graphs and info page
if selected == 'Metrics':
    metrics_page()


# Classification page
elif selected == 'Classify':
    classify_page()
    