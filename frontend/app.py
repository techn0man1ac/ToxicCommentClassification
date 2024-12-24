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
selected = option_menu(menu_title=None, options=["Home", "Classify"],
                       menu_icon="cast", default_index=0, icons=['house', 'play'],
                       orientation="horizontal")

# Home page
if selected == 'Home':
    st.write('To be added...')

# Classification page
elif selected == 'Classify':

    # Toxicity labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Model selectbox
    model_choice = st.selectbox('Choose your model', ['model1'])

    # User's comment input
    user_comment = st.text_area('Enter your comment here')

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