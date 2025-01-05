# Toxic Comment Classification system by "Team 16.6" 🛡️

![Team 16.6 logo](https://raw.githubusercontent.com/techn0man1ac/ToxicCommentClassification/refs/heads/main/frontend/imgs/team16_6_Logo.png)

In the modern world of social media, there is a significant problem of toxicity in online comments, which creates a negative environment for communication. From abuse to insults, this can lead to a cessation of the exchange of thoughts and ideas among users. This project aims to develop a model capable of identifying and classifying different levels of toxicity in comments, using the power of BERT(Bidirectional Encoder Representations from Transformers for text analysis.

This project aims to develop a machine learning model that can effectively classify different levels of toxicity in online comments. We use advanced technologies such as BERT to analyze text and create a system that will help moderators and users create healthier and safer social media environments.

# 🛠️ Technologies

- 🐍 [Python](https://www.python.org/): The application was developed in the [Python 3.11.8](https://www.python.org/downloads/release/python-3118/) programming language
- 🤗 [Transformers](https://huggingface.co/docs/transformers/index): A library that provides access to BERT and other advanced machine learning models
- 🔥 [PyTorch](https://pytorch.org/): Libraries for working with deep learning
- 📖 [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)): A text analysis model used to produce contextualized word embeddings
- ☁️ [Kaggle](https://www.kaggle.com/): To save time, we used cloud computing to train the models
- 🌐 [Streamlit](https://streamlit.io/): To develop the user interface, used the Streamlit package in the frontend
- 🐳 [Docker](https://www.docker.com/): A platform for building, deploying, and managing containerized applications

# 📊 Dataset(EDA)

To train the machine learning models, we used [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataset. 
The dataset have types of toxicity:
- Toxic 🗯️  
- Severe Toxic 🤬  
- Obscene 🚫  
- Threat ☠️  
- Insult 🗣️  
- Identity Hate 👤💔 

# 🖥 Data Science

The primary datasets (`train.csv`, `test.csv`, and `sample_submission.csv`) are loaded into [Pandas](https://pandas.pydata.org/) `DataFrames`. 
After that make [Exploratory Data Analysis](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/Data_science/data_science.ipynb) of dataframes and obtained the following results:

As you can seen from the data analysis, there is an `imbalance of classes` in the ratio of 1 to 10 (toxic/non-toxic). 

![Data toxic distribution ](https://raw.githubusercontent.com/techn0man1ac/ToxicCommentClassification/refs/heads/main/IMGs/dataToxicDistribution.png)

Distribution of classes:

| Class          | Count   | Percentage |
|----------------|---------|------------|
| toxic          | 15294   | 8.57%      |
| severe_toxic   | 1595    | 0.89%      |
| obscene        | 8449    | 4.73%      |
| threat         | 478     | 0.27%      |
| insult         | 7877    | 4.41%      |
| identity_hate  | 1405    | 0.79%      |
| **Non-toxic**     | **143346**  | **80.33%**     |
| Total comments | 178444 |      |

Here is a visualization of the data from the dataset research. Dataset in bargraph representation:

![Dataset in bar graph format](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/IMGs/dataSetGraphic0.png)

Dataset in pie representation:

![Dataset in pie format](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/IMGs/dataSetGraphic1.png)

Graphs show basic information about the dataset to understand the size and types of columns. Such a ratio in the data will have a very negative impact on the model's prediction accuracy.

# 📅 Data processing

| Class          | Count  | Percentage |
|----------------|--------|------------|
| toxic          | 15294  | 22.53%     |
| severe_toxic   | 15500  | 22.84%     |
| obscene        | 15654  | 23.06%     |
| threat         | 12732  | 18.76%     |
| insult         | 15088  | 22.23%     |
| identity_hate  | 13816  | 20.35%     |
| Non-toxic      | 16225  | 23.90%     |

The [balance of classes has been adjusted](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/Data_science/preprocessing_data/preprocessing_data.ipynb), and the balance of non-toxic categories has been changed for each class, and now the balance is 50/50 for each category. After that make cleaning process ensures compatibility with machine learning models such as BERT.

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
  - ✅ Accuracy: 0.95  
  - ✅ Precision: 0.97  
  - ✅ Recall: 0.96  

- **Model Specifications**:  
  - Vocabulary Size: 30,522  
  - Hidden Size: 768  
  - Attention Heads: 12  
  - Hidden Layers: 12  
  - Total Parameters: 110M  
  - Maximum Sequence Length: 512  
  - Pre-trained Tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).  

## ALBERT ֎🇦🇮
...

## DistilBERT ֎🇦🇮
...

# 💻 How to install
...

# 🚀 How to use(Front End)
...

# 🎯 Mission 

The mission of our project is to create a reliable and accurate machine learning model that can effectively classify different levels of toxicity in online comments. We plan to use advanced technologies to analyze text and create a system that will help moderators and users create healthier and safer social media environments.

# 🌟 Vision

Our vision is to make online communication safe and comfortable for everyone. We want to build a system that not only can detect toxic comments, but also helps to understand the context and tries to reduce the number of such messages. We want to create a tool that will be used not only by moderators, but also by every user to provide a safe environment for the exchange of thoughts and ideas.

# 📜 Licence

This project is a group work published under the [MIT license](https://github.com/techn0man1ac/ToxicCommentClassification/blob/main/LICENSE) , and all project contributors are listed in the license text.

# 👏 Acknowledgments

🎓 This project was developed by a team of professionals as a graduation thesis of the [Python Data Science and Machine Learning](https://goit.global/ua/courses/python-ds/) course 🎯 .

🎉 **Thank you for exploring our project! Together, we can make online spaces healthier and more respectful.**
