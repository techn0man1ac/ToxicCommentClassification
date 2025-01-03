import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model_func import predict_toxicity


# Toxicity labels
label_list = ['toxic', 'severe_toxic', 'obscene', 
				'threat', 'insult', 'identity_hate']

# Model directories
bert_model_dir = "saved_models/bert"
albert_model_dir = "saved_models/albert"
distilbert_model_dir = "saved_models/distilbert"


def classify_page():
    # Model selectbox
    model_choice_classify = st.selectbox('Choose your model', ['BERT', 'ALBERT', 'DISTILBERT'])

    # Choice of model
    if model_choice_classify == 'BERT':
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
        model, tokenizer = bert_model, bert_tokenizer

    elif model_choice_classify == 'ALBERT':
        albert_model = AlbertForSequenceClassification.from_pretrained(albert_model_dir)
        albert_tokenizer = AlbertTokenizer.from_pretrained(albert_model_dir)
        model, tokenizer = albert_model, albert_tokenizer

    else:
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_dir)
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_dir)
        model, tokenizer = distilbert_model, distilbert_tokenizer

    # User's comment input
    user_comment = st.text_area('Enter your comment here')

    # User's text file upload
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])

    if uploaded_file is not None:
        user_comment = uploaded_file.read().decode("utf-8")

    user_comment = [user_comment]

    # Detailed toxicity checkbox
    detailed_classification = st.checkbox('Display detailed toxicity')

    classify = st.button('Classify')
    
    if classify:
        # Model prediciton
        prediction = predict_toxicity(user_comment, model, tokenizer)

        # Comment is toxic when as least one of the toxicity classes exceedes 0.5
        is_toxic = True if True in (prediction > 0.5) else False

        st.write('The overall comment is:', 'toxic.' if is_toxic == True else 'not toxic.')

        # Display bar chart
        if detailed_classification:
            data = pd.DataFrame({'Label': label_list, 'Value': prediction})
            data.set_index('Label', inplace=True)
            st.bar_chart(data['Value'])
            