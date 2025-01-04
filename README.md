# Toxic Comment Classification system by "Team 16.6"

![Team 16.6 logo](https://raw.githubusercontent.com/techn0man1ac/ToxicCommentClassification/refs/heads/main/frontend/imgs/team16_6_Logo.png)

In the modern world of social media, there is a significant problem of toxicity in online comments, which creates a negative environment for communication. From abuse to insults, this can lead to a cessation of the exchange of thoughts and ideas among users. This project aims to develop a model capable of identifying and classifying different levels of toxicity in comments, using the power of BERT(Bidirectional Encoder Representations from Transformers for text analysis.

This project aims to develop a machine learning model that can effectively classify different levels of toxicity in online comments. We use advanced technologies such as BERT to analyze text and create a system that will help moderators and users create healthier and safer social media environments.

# Technologies

- [BERT](https://huggingface.co/docs/transformers/model_doc/bert): A text analysis model used to produce contextualized word embeddings.
- [PyTorch](https://pytorch.org/): Libraries for working with deep learning.
- [Transformers](https://huggingface.co/docs/transformers/index): A library that provides access to BERT and other advanced machine learning models.
- [Docker](https://www.docker.com/): A platform for building, deploying, and managing containerized applications.

# Dataset(EDA)

To train the machine learning models, we used [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataset. 
The dataset have types of toxicity:
- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

After [Exploratory Data Analysis](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Data_science), we found that the dataset had an `imbalance of classes`, and this had a bad impact on model training.

# Data processing(data science)
...

# Machine learning(back end)

To solve the problem of the application, we have chosen 3 popular architectures, such as [BERT](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Backend/Models/Model_0_bert-base-uncased), [ALBERT](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Backend/Models/Model_1_albert), [DistilBERT](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Backend/Models/Model_2_distilbert).

## BERT

This project demonstrates toxic comment classification using the [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) model from the BERT family.

### 1. **Toxic Comment Classification with BERT**
- Utilized the `bert-base-uncased` model with PyTorch for flexibility and ease of use.  
- Seamlessly integrated with Hugging Face Transformers.  
- Training accelerated by ~30x using a GPU, efficiently handling BERT’s computational demands.

### 2. **Dataset Balancing**
- Addressed dataset imbalance (90% non-toxic, 10% toxic) using oversampling with `sklearn`.  
- Ensured rare toxic categories received equal attention by balancing class distributions.  
- Improved model performance in recognizing rare toxic classes.

### 3. **Key Techniques**
- **Tokenization**: Preprocessed data tokenized using `BertTokenizer`.  
- **Loss Function**: Used `BCEWithLogitsLoss` with weighted loss for rare class emphasis.  
- **Gradient Clipping**: Optimized training stability with gradient clipping (`max_norm`).  
- **Hyperparameter Tuning**: Tuned batch size, learning rate, and epochs using Optuna.  

### 4. **Threshold Optimization**
- Used `itertools.product` to find optimal thresholds for each class.  
- Improved recall and F1-score (by 1-1.5%) for better multi-label classification.

### 5. **Performance and Key Model Details**
- **Validation Metrics**:  
  - Accuracy: 0.95  
  - Precision: 0.97  
  - Recall: 0.96  

- **Model Specifications**:  
  - Vocabulary Size: 30,522  
  - Hidden Size: 768  
  - Attention Heads: 12  
  - Hidden Layers: 12  
  - Total Parameters: 110M  
  - Maximum Sequence Length: 512  
  - Pre-trained Tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).  

## ALBERT
...

## DistilBERT
...

# How to install
...

# How to use(front end)
...

# Mission 

The mission of our project is to create a reliable and accurate machine learning model that can effectively classify different levels of toxicity in online comments. We plan to use advanced technologies such as BERT (Bidirectional Encoder Representations from Transformers) to analyze text and create a system that will help moderators and users create healthier and safer social media environments.

# Vision

Our vision is to make online communication safe and comfortable for everyone. We want to build a system that not only can detect toxic comments, but also helps to understand the context and tries to reduce the number of such messages. We want to create a tool that will be used not only by moderators, but also by every user to provide a safe environment for the exchange of thoughts and ideas.

# Licence

This project is a group work published under the [MIT license](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/LICENSE) , and all project contributors are listed in the license text.
