import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model_func import predict_toxicity
from streamlit_option_menu import option_menu
import plotly.figure_factory as ff
import ast


# Toxicity labels
label_list = ['toxic', 'severe_toxic', 'obscene', 
				'threat', 'insult', 'identity_hate']

# Model directory
bert_model_dir = "saved_models/bert"
albert_model_dir = "saved_models/albert"
distilbert_model_dir = "saved_models/distilbert"

# Create two columns
col1, col2 = st.columns([1, 5])

# Image column
with col1:
    img = Image.open("imgs/team16_6_Logo.png")  # Use the relative path or absolute path
    st.image(img, use_container_width=True)


# Title column
with col2:
    st.title('Toxic Comment Classification System by "Team 16.6"')


# Option menu
selected = option_menu(menu_title=None, options=["Home", "About", 'Metrics', "Classify"],
                       menu_icon="cast", default_index=0, icons=['house', 'people', 'clipboard-data', 'play'],
                       orientation="horizontal")


# Home page
if selected == 'Home':
    st.title('Welcome!')
    st.write('Use this app to analyse texts for toxicity.')


# Team page
if selected == 'About':
    st.title('Our Team:')

    team = [
        {"name": "Serhii Trush", "role": "Team Lead", "github": "https://github.com/techn0man1ac"},
        {"name": "Oleksandr Kovalenko", "role": "SCRUM Master", "github": "https://github.com/AlexandrSergeevichKovalenko"},
        {"name": "Aliona Mishchenko", "role": "Data Scientist", "github": "https://github.com/Alena-Mishchenko"},
        {"name": "Ivan Shkvir", "role": "Backend Developer", "github": "https://github.com/IvanShkvyr"},
        {"name": "Oleksii Yeromenko", "role": "Frontend Developer", "github": "https://github.com/oleksii-yer"},
        {"name": "Polina Mamchur", "role": "Creative Director", "github": "https://github.com/polinamamchur"}
    ]


    # Display team members in columns
    for i in range(0, len(team), 2):  # Display 2 members per row
        cols = st.columns(2)
        for col, member in zip(cols, team[i:i+2]):
            with col:
                # Display member name and role
                st.markdown(f"### [{member['name']}]({member['github']})")
                st.markdown(f"*{member['role']}*")

    st.markdown("---")

    st.title('What is this app about?')

    st.write('This app can be used to classify any given comment or text based on its toxicity.')
    st.write('Three BERT-based models in total have been trained and developed.')


# Metric graphs and info page
if selected == 'Metrics':
    model_choice = st.selectbox('Choose your model', ['BERT', 'ALBERT', 'DISTILBERT'])

    metrics_df = pd.read_csv('data/bert_metrics.csv')
    conf_matrix_df = pd.read_csv('data/confusion_matrix.csv')


    # Assuming `metrics_df` contains your metrics data
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Value': [0.9382, 0.9607, 0.9092]
    })

    # Displaying model metrics as a table
    st.write('### Model Metrics:')
    metrics_df_reset = metrics_df.reset_index(drop=True)

    # Display the table without the index column
    st.dataframe(metrics_df_reset, hide_index=True)

    for i, row in conf_matrix_df.iterrows():

        matrix = [[row['FN'], row['TP']], [row['TN'], row['FP']]]
        columns_x = [f'Predicted not {label_list[i]}', f'Predicted {label_list[i]}']  # Example classes
        columns_y = [f'{label_list[i]}', f'Not {label_list[i]}']


        # Create a Plotly figure with enhancements
        fig = ff.create_annotated_heatmap(
            z=matrix,
            x=columns_x,
            y=columns_y,
            colorscale='Blues',
            showscale=True,  # Add the color scale bar
            colorbar_title='Count',  # Title for the color scale
            colorbar_tickprefix=' ',  # Optional: clean up tick labels if needed
        )

        # Customize layout for better visuals
        fig.update_layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted Labels'),
            yaxis=dict(title='True Labels'),
            autosize=True,
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)


# Classification page
elif selected == 'Classify':

    # Завантаження збереженої моделі та токенізатора
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

    albert_model = AlbertForSequenceClassification.from_pretrained(albert_model_dir)
    albert_tokenizer = AlbertTokenizer.from_pretrained(albert_model_dir)

    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_dir)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_dir)

    bert_model.to('cpu')

    # Model selectbox
    model_choice = st.selectbox('Choose your model', ['BERT', 'ALBERT', 'DISTILBERT'])

    if model_choice == 'BERT':
        model, tokenizer = bert_model, bert_tokenizer
    elif model_choice == 'ALBERT':
        model, tokenizer = albert_model, albert_tokenizer
    else:
        model, tokenizer = distilbert_model, distilbert_tokenizer

    # User's comment input
    user_comment = st.text_area('Enter your comment here')

    # User's text file upload
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])

    if uploaded_file is not None:
        user_comment = uploaded_file.read().decode("utf-8")

    user_comment = [user_comment]

    # Toxicity probabilities checkbox
    detailed_classification = st.checkbox('Display detailed toxicity')

    classify = st.button('Classify')
    
    if classify:

        # Model prediciton
        prediction = predict_toxicity(user_comment, model, tokenizer)

        is_toxic = True if True in (prediction > 0.5) else False

        st.write('The overall comment is:', 'toxic.' if is_toxic == True else 'not toxic.')

        # Display bar chart
        if detailed_classification:
            data = pd.DataFrame({'Label': label_list, 'Value': prediction})
            data.set_index('Label', inplace=True)
            st.bar_chart(data['Value'])