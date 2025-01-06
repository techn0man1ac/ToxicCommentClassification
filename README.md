# Toxic Comment Classification system by "Team 16.6" 🛡️

![Team 16.6 logo](https://raw.githubusercontent.com/techn0man1ac/ToxicCommentClassification/refs/heads/main/frontend/imgs/team16_6_Logo.png)

In the modern era of social media, toxicity in online comments poses a significant challenge, creating a negative atmosphere for communication. From abuse to insults, toxic behavior discourages the free exchange of thoughts and ideas among users.

This project seeks to address this issue by developing a machine learning model to identify and classify varying levels of toxicity in comments. Leveraging the power of BERT (Bidirectional Encoder Representations from Transformers), this system aims to:

- Analyze text for signs of toxicity
- Classify toxicity levels effectively
- Support moderators and users in fostering healthier and safer online communities
- By implementing this technology, the project strives to make social media a more inclusive and positive space for interaction.

# 🛠️ Technologies

- 🐍 [Python](https://www.python.org/): The application was developed in the [Python 3.11.8](https://www.python.org/downloads/release/python-3118/) programming language
- 🤗 [Transformers](https://huggingface.co/docs/transformers/index): A library that provides access to BERT and other advanced machine learning models
- 🔥 [PyTorch](https://pytorch.org/): Libraries for working with deep learning
- 📖 [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)): A text analysis model used to produce contextualized word embeddings
- ☁️ [Kaggle](https://www.kaggle.com/): To save time, we used cloud computing to train the models
- 🌐 [Streamlit](https://streamlit.io/): To develop the user interface, used the Streamlit package in the frontend
- 🐳 [Docker](https://www.docker.com/): A platform for building, deploying, and managing containerized applications

# 🖥 Data Science

Work was performed on dataset research and data processing to train machine learning models. 

## 📊 Dataset(EDA)

To train the machine learning models, we used [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataset. 
The dataset have types of toxicity:
- Toxic 
- Severe Toxic 
- Obscene
- Threat
- Insult
- Identity Hate

The primary datasets (`train.csv`, `test.csv`, and `sample_submission.csv`) are loaded into [Pandas](https://pandas.pydata.org/) `DataFrames`. 
After that make [Exploratory Data Analysis](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/Data_science/data_science.ipynb) of dataframes and obtained the following results:

![Data toxic distribution ](https://raw.githubusercontent.com/techn0man1ac/ToxicCommentClassification/refs/heads/main/IMGs/dataToxicDistribution.png)

As you can seen from the data analysis, there is an `imbalance of classes` in the ratio of 1 to 10 (toxic/non-toxic). 

Distribution of classes:

| Class          | Count       | Percentage      |
|----------------|-------------|-----------------|
| Toxic          | 15,294      | 9.58%           |
| Severe Toxic   | 1,595       | 1.00%           |
| Obscene        | 8,449       | 5.29%           |
| Threat         | 478         | 0.30%           |
| Insult         | 7,877       | 4.94%           |
| Identity Hate  | 1,405       | 0.88%           |
| Non-toxic      | **143,346** | **89.83%**      |
| Total comments | 178444      |                 |

As you can see, this table shows that there is multiclassing in the data, the data of one category can belong to another category.

Here is a visualization of the data from the dataset research. Dataset in bargraph representation:

![Dataset in bar graph format](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/IMGs/dataSetGraphic0.png)

Dataset in pie representation:

![Dataset in pie format](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/IMGs/dataSetGraphic1.png)

Graphs show basic information about the dataset to understand the size and types of columns. Such a ratio in the data will have a very negative impact on the model's prediction accuracy.

## 📅 Data processing

| Class          | Count  | Percentage |
|----------------|--------|------------|
| toxic          | 15294  | 22.53%     |
| severe_toxic   | 15500  | 22.84%     |
| obscene        | 15654  | 23.06%     |
| threat         | 12732  | 18.76%     |
| insult         | 15088  | 22.23%     |
| identity_hate  | 13816  | 20.35%     |
| Non-toxic      | 16225  | 23.90%     |

[Correction](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/Data_science/preprocessing_data/preprocessing_data.ipynb) of general amount of toxic and non-toxic classes to reach 50% / 50% balance between them in comparison to the original data where general amount of all toxic classes has a share of 10% in comparison to non-toxic, which had 90% of the share of the original data set.
These all steps would ensure toxic category in general as well as each toxic class receives equal model attention during learning process.

# ⚙️ Machine learning(Back End)

To solve the problem of the application, we have chosen 3 popular architectures, such as [BERT](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Backend/Models/Model_0_bert-base-uncased), [ALBERT](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Backend/Models/Model_1_albert), [DistilBERT](https://github.com/techn0man1ac/ToxicCommentClassification/tree/main/Backend/Models/Model_2_distilbert).

## BERT ֎🇦🇮

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
  - Accuracy: 0.95 ✅
  - Precision: 0.97 ✅
  - Recall: 0.96 ✅  

- **Model Specifications**:  
  - Vocabulary Size: 30,522  
  - Hidden Size: 768  
  - Attention Heads: 12  
  - Hidden Layers: 12  
  - Total Parameters: 110M  
  - Maximum Sequence Length: 512(in this case use 128 tokens)
  - Pre-trained Tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).  

## ALBERT ֎🇦🇮
...

## DistilBERT ֎🇦🇮
 
This project demonstrates toxic comment classification using the [DistilBertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/distilbert) model, a lightweight and efficient version of BERT.

### 1. Using PyTorch  
- Selected for its flexibility, ease of use, and strong community support.  
- Seamlessly integrated with Hugging Face Transformers.  

### 2. Dataset Balancing  
- Addressed dataset imbalance (90% non-toxic, 10% toxic) using `sklearn.utils.resample`.  
- Applied stratified splitting for training and test datasets.  
- Oversampled rare toxic classes, improving model recognition of all categories.  

### 3. Key Techniques  
- **Tokenization**: Preprocessed data with `DistilBertTokenizer`.  
- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).  
- **Hyperparameter Tuning**: Optimized batch size (`16`), learning rate (`2e-5`), and epochs (`3`) with Optuna.  

### 4. Accelerated Training  
- Utilized GPU for training, achieving a ~30x speedup over CPU.

### 5. Threshold Optimization  
- Used `itertools.product` to determine optimal thresholds for each class.  
- Improved recall and F1-score for multi-label classification. 

### 6. **Performance and Key Model Details** 
- **Validation Metrics**:   
  - Accuracy: 0.92 ✅
  - Precision: 0.79 ✅
  - Recall: 0.78 ✅

- **Model Specifications**:  
  - Vocabulary Size: 30522  
  - Hidden Size: 768 
  - Attention Heads: 12 
  - Hidden Layers: 6  
  - Total Parameters: 66M  
  - Maximum Sequence Length: 512(in this case use 128 tokens)
  - Pre-trained Tasks: Masked Language Modeling (MLM).  

# 💻 How to install

There are two ways to install the application on your computer:

## Simple 😎

...

## Like are pro 💪

...

# 🚀 How to use(Front End)

...

# 🤝 Team

The project was divided into tasks and assigned to the following roles:
...

# 🎯 Mission 

The mission of our project is to create a reliable and accurate machine learning model that can effectively classify different levels of toxicity in online comments. We plan to use advanced technologies to analyze text and create a system that will help moderators and users create healthier and safer social media environments.

# 🌟 Vision

Our vision is to make online communication safe and comfortable for everyone. We want to build a system that not only can detect toxic comments, but also helps to understand the context and tries to reduce the number of such messages. We want to create a tool that will be used not only by moderators, but also by every user to provide a safe environment for the exchange of thoughts and ideas.

# 📜 Licence

This project is a group work published under the [MIT license](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/LICENSE) , and all project contributors are listed in the license text.

# 👏 Acknowledgments

This project was developed by a team of professionals as a graduation thesis of the [GoIT](https://goit.global/) **Python Data Science and Machine Learning** course.

Thank you for exploring our project! Together, we can make online spaces healthier and more respectful.
