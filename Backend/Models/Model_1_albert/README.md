# Toxic Comment Classification using AlBERT (albert-base-v2)  
This project demonstrates how to classify toxic comments using the  
AlbertForSequenceClassification model from the BERT family. This model is a  
lightweight and efficient version of BERT (AlBERT), designed to reduce the number  
of parameters while maintaining high performance and accuracy. Below are the  
key highlights of the implementation:

## Key Highlights

### 1. Using PyTorch
The PyTorch library was chosen for this project due to:
- Its flexibility and ease of use, making it ideal for deep learning experiments.
- Strong community support and a wide range of available pre-trained models.
- Seamless integration with the Hugging Face `transformers` library.

### 2. Dataset Imbalance and Oversampling
The original dataset was highly imbalanced:
- **90%** of the comments were non-toxic, while only **10%** were toxic.
- Among toxic comments, there was significant class imbalance across different toxic categories.

#### Solution:
- The classes were balanced using the resample function from the sklearn.utils library.  
- The number of toxic comments was equalized with the number of non-toxic comments.  
- Stratified splitting was applied to the training and test datasets.  
- Oversampling of rare classes ensured that the model paid equal attention to all categories.  


#### Problem Addressed:
Before oversampling, the model struggled to classify rare toxic classes and often  
ignored them. Balancing the dataset mitigated this issue, improving the model's  
ability to recognize rare classes.

### 3. Tokenization
After data preprocessing, the new dataset was tokenized using `AlbertTokenizer`.  
This step prepared the data for input into the AlBERT model.

### 4. Accelerated Training with GPU
The training process utilized a GPU, which:
- Improved the training speed by approximately **30 times** compared to using a CPU.
- Enabled efficient handling of the computational demands of the AlBERT model.

### 5. Loss Function: BCEWithLogitsLoss
The loss function used was **Binary Cross-Entropy with Logits (BCEWithLogitsLoss)**.  
This loss function is well-suited for multi-label classification because:
- It combines the sigmoid activation function with binary cross-entropy loss.
- It measures the difference between predicted probabilities and true binary labels.

### 6. Hyperparameter Tuning with Optuna
The **Optuna** library was used to optimize key hyperparameters:
- **Batch size**: `8`
- **Learning rate**: `2e-5`
- **Number of epochs**: `3`

### 7. Optimizing Thresholds for Multi-Label Classification
The `product` function from Python's `itertools` library was used to find the  
best thresholds for each class. These thresholds determine when a class is  
considered positive.

### 8. Model Performance on Validation Data
Using all the techniques mentioned above, the model achieved the following metrics on the validation dataset:
- **Accuracy**: `0.92`
- **Precision**: `0.85`
- **Recall**: `0.69`

---

## Key Parameters of `bert-base-uncased`

1. **Vocabulary Size**: `30000`  
 Number of unique tokens in the model's vocabulary.

2. **Hidden Size**: `768`  
 Dimensionality of the token embeddings and hidden states.

3. **Number of Attention Heads**: `12`  
 Parallel attention mechanisms per layer.

4. **Number of Hidden Layers**: `12`  
 Transformer encoder layers in the model.

5. **Intermediate Size**: `4096`  
 Size of the feed-forward layers.

6. **Maximum Sequence Length**: `512`  
 Maximum number of tokens per input sequence.

7. **Dropout Rate**: `0.1`  
 Dropout applied to prevent overfitting.

8. **Number of Parameters**: `11 million`  
 Total trainable parameters.

9. **Uncased**: `True`  
 Input text is case-insensitive (all converted to lowercase).

10. **Pre-trained Tasks**: Masked Language Modeling (MLM)
