# Toxic Comment Classification Challenge

All research is organized around the dataset [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/).

# Project Overview  

In this project, my primary goal was to analyze and preprocess toxic comments to create a reliable dataset for training a BERT model.  

## Data Analysis   

- **Loading and Analyzing DataFrame Structure**  
  Initial inspection and analysis of the dataset to understand its structure and content.
  
- **Creating a DataFrame of the Top 20 Toxic Words**  
  For each toxic comment class, a DataFrame was generated to highlight the 20 most frequently occurring toxic words.  

## Preprocessing Data  

- **Comment Preprocessing for BERT Models**  
  Cleaning and preparing comment text for input into BERT models to ensure effective learning.
   
- **Balancing Toxic and Non-Toxic Comments**  
  Using `RandomUnderSampler` to eliminate the imbalance between all toxic comments and non-toxic comments, ensuring a more even distribution.
    
- **Class Imbalance Adjustment Through Augmentation**  
  Augmenting underrepresented toxic classes using **`SynonymAug`** to ensure that the number of comments is balanced across all classes. This augmentation specifically targets the three least represented toxic classes.
  
- **Tokenizing the DataFrame for BERT Models**  
  Tokenizing comments to prepare them for model training and evaluation using BERT.  
