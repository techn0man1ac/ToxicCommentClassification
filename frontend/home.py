import streamlit as st


# Path to logo files
logo_paths = [
    "imgs/GoIT.png",
    "imgs/python.png",
    "imgs/pyTorch.png",
    "imgs/nvidia.png",
    "imgs/streamlit.png",
    "imgs/kaggle.png",
    "imgs/huggingface.png",
    "imgs/docker.png",
    "imgs/google.png",
]


def home_page():
    st.title('Welcome!')
    # Project description
    st.markdown('''In the modern world of social media, there is a significant problem of toxicity in online comments, 
    which creates a negative environment for communication. From abuse to insults, 
    this can lead to a cessation of the exchange of thoughts and ideas among users. 
    This project aims to develop a model capable of identifying and classifying different 
    levels of toxicity in comments, using the power of 
    [BERT(Bidirectional Encoder Representations from Transformers)](https://en.wikipedia.org/wiki/BERT_(language_model)) 
    for text analysis.

# Project Description

This project aims to develop a machine learning model that can effectively classify different levels of toxicity in 
online comments. We use advanced technologies such as BERT (Bidirectional Encoder Representations from Transformers) 
for that purpose.

# Technologies

- BERT (Bidirectional Encoder Representations from Transformers): A text analysis model used to produce contextualized 
word embeddings.
- **PyTorch**: Libraries for working with deep learning.
- Transformers: A library that provides access to BERT and other advanced machine learning models.
- Docker: A platform for building, deploying, and managing containerized applications.

# Dataset

We use [Toxic Comment Classification Challenge]() dataset for training machine learning models. 
The dataset have types of toxicity:
- Toxic
- Severe_toxic
- Obscene
- Threat
- Insult
- Identity_hate

# Mission 
The mission of our project is to create a reliable and accurate machine learning model that can effectively classify 
different levels of toxicity in online comments. We have used advanced technologies such as BERT 
(Bidirectional Encoder Representations from Transformers) to analyze text and create a system that will help moderators 
and users create healthier and safer social media environments.

# Vision
Our vision is to make online communication safe and comfortable for everyone. We want to build a system that not only 
can detect toxic comments, but also helps to understand the context and tries to reduce the number of such messages. 
We want to create a tool that will be used not only by moderators, but also by every user to provide a safe environment 
for the exchange of thoughts and ideas.

# Licence

This project is a group work published under the 
[MIT license](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/LICENSE), and all project 
contributors are listed in the license text.'''
    )

    st.markdown('---')
    st.markdown('# Powered By')

    # Display logos in a row
    columns = st.columns(len(logo_paths))
    for col, logo_path in zip(columns, logo_paths):
        with col:
            st.image(logo_path, width=2000)
            