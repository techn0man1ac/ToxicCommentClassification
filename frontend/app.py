import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.figure_factory as ff

# Toxicity labels
label_list = ['toxic', 'severe_toxic', 'obscene', 
				'threat', 'insult', 'identity_hate']

def predict_toxicity(texts, model, tokenizer):
    model.eval()
    
    # Токенізація введених текстів
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    
    # Передбачення
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.sigmoid(logits) 
        
    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    
    return predicted_labels[0]


# Create two columns
col1, col2 = st.columns([1, 4])

# Image column
with col1:
    img = Image.open("imgs/team16_6_Logo.png")  # Use the relative path or absolute path
    st.image(img, use_container_width=True)


# Title column
with col2:
    st.title("Group 16.6. Comment classification using BERT")


# Option menu
selected = option_menu(menu_title=None, options=["Home", "Team", 'Metrics', "Classify"],
                       menu_icon="cast", default_index=0, icons=['house', 'people', 'clipboard-data', 'play'],
                       orientation="horizontal")


# Home page
if selected == 'Home':
    st.title('Welcome!')
    st.write('Use this app to analyse texts for toxicity.')


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


# Metric graphs and info page
if selected == 'Metrics':
    
    # Assuming you have the `y_true` and `y_pred` (ground truth and predictions)
    y_true = [0, 1, 0, 1, 0, 1]  # Example true values
    y_pred = [0, 0, 0, 1, 1, 1]  # Example predicted values
    columns = ['0', '1']  # Example classes

    # Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Create a Plotly figure with enhancements
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=columns,
        y=columns,
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
        xaxis_tickangle=-45,  # Optional: Angle the x-axis ticks for better readability
        yaxis_tickangle=45,   # Optional: Angle the y-axis ticks for better readability
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Classification page
elif selected == 'Classify':

    # Шлях до збереженої моделі
    output_dir = "saved_model"

    # Завантаження збереженої моделі та токенізатора
    bert_model = BertForSequenceClassification.from_pretrained(output_dir)
    bert_tokenizer = BertTokenizer.from_pretrained(output_dir)

    bert_model.to('cpu')

    # Model selectbox
    model_choice = st.selectbox('Choose your model', ['BERT'])

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
        prediction = predict_toxicity(user_comment, bert_model, bert_tokenizer)

        is_toxic = True if True in (prediction > 0.5) else False

        st.write('The overall comment is:', 'toxic.' if is_toxic == True else 'not toxic.')

        # Display bar chart
        if detailed_classification:
            data = pd.DataFrame({'Label': label_list, 'Value': prediction})
            data.set_index('Label', inplace=True)
            st.bar_chart(data['Value'])