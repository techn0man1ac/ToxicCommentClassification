import streamlit as st
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu

# Model load
# model1 = load_model('some_model.h5')

# Create two columns
col1, col2 = st.columns([1, 4])

# Image column
with col1:
    img = Image.open("imgs/team16_6_Logo.png")  # Use the relative path or absolute path
    st.image(img, use_container_width=True)


# Title column
with col2:
    st.title("Group 4. Comment classification using BERT")


# Option menu
selected = option_menu(menu_title=None, options=["Home", "Team", "Classify"],
                       menu_icon="cast", default_index=0, icons=['house', 'people', 'play'],
                       orientation="horizontal")


# Home page
if selected == 'Home':
    st.title('Welcome!')
    st.write('Use this app to analyse texts for toxicity.')
    to_classify = st.button('Go to Classify')
    if to_classify:
        selected = 'Classify'


# Team page
if selected == 'Team':
    st.title('Our Team')

    st.markdown("---")

    team = [
        {"name": "Serhii Trush", "role": "Team Lead"},
        {"name": "Oleksandr Kovalenko", "role": "SCRUM Master"},
        {"name": "Aliona Mishchenko", "role": "Data Scientist"},
        {"name": "Ivan Shkvir", "role": "Backend Developer"},
        {"name": "Oleksii Yeromenko", "role": "Frontend Developer"},
        {"name": "Polina Mamchur", "role": "Creative Director"}
    ]

    # Display team members in columns
    for i in range(0, len(team), 2):  # Display 2 members per row
        cols = st.columns(2)
        for col, member in zip(cols, team[i:i+2]):
            with col:
                # Display member name and role
                st.markdown(f"### {member['name']}")
                st.markdown(f"*{member['role']}*")


# Classification page
elif selected == 'Classify':

    # Toxicity labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Model selectbox
    model_choice = st.selectbox('Choose your model', ['model1'])

    # User's comment input
    user_comment = st.text_area('Enter your comment here')

    # User's text file upload
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])

    if uploaded_file is not None:
        user_comment = uploaded_file.read().decode("utf-8")

    # Toxicity probabilities checkbox
    detailed_classification = st.checkbox('Display toxicity probabilities')

    classify = st.button('Classify')
    
    if classify:

        # Model prediciton
        # result = model_choice.predict(user_comment)

        # Example array
        result = np.array([1, 2, 3, 4, 5, 6], dtype= float)

        is_toxic = True if True in (result > 0.1) else False

        st.write('The comment is:', 'toxic.' if is_toxic == True else 'not toxic.')

        # Display bar chart
        if detailed_classification:
            data = pd.DataFrame({'Label': labels, 'Value': result})
            data.set_index('Label', inplace=True)
            st.bar_chart(data['Value'])