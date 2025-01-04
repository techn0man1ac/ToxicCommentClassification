# Toxic Comment Classification using BERT (`bert-base-uncased`)

This project demonstrates how to classify toxic comments using the `bert-base-uncased` model from the BERT family. Below are the key highlights of the implementation:

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
- The classes were balanced using the `resample` function from `sklearn.utils`.
- The number of toxic comments was equalized to the number of non-toxic comments.
- Oversampling rare classes ensured the model paid equal attention to all categories.

#### Problem Addressed:
Before oversampling, the model struggled to classify rare toxic classes and often ignored them. Balancing the dataset mitigated this issue, improving the model's ability to recognize rare classes.

### 3. Tokenization
After data preprocessing, the new dataset was tokenized using `BertTokenizer`. This step prepared the data for input into the BERT model.

### 4. Accelerated Training with GPU
The training process utilized a GPU, which:
- Improved the training speed by approximately **30 times** compared to using a CPU.
- Enabled efficient handling of the computational demands of the BERT model.

### 5. Loss Function: BCEWithLogitsLoss
The loss function used was **Binary Cross-Entropy with Logits (BCEWithLogitsLoss)**. This loss function is well-suited for multi-label classification because:
- It combines the sigmoid activation function with binary cross-entropy loss.
- It measures the difference between predicted probabilities and true binary labels.

#### Weighted Loss:
Class weights were computed as the inverse frequency of each class. Rare classes received higher weights, emphasizing their importance and preventing the model from ignoring them.

### 6. Hyperparameter Tuning with Optuna
The **Optuna** library was used to optimize key hyperparameters:
- **Batch size**: `32`
- **Learning rate**: `3.50425473e-5`
- **Number of epochs**: `2`
- **Max norm**: `0.714110939` (used for gradient clipping to stabilize training)

#### Gradient Clipping:
- Gradient clipping constrains the norm (magnitude) of gradients to prevent:
  - **Exploding gradients**: Large gradients that disrupt the training process.
  - **Numerical instability**: Errors caused by overly large weight updates.
- By including `max_norm` in the hyperparameter search, the stability of the training process was optimized alongside the learning rate and batch size.

### 7. Optimizing Thresholds for Multi-Label Classification
The `product` function from Python's `itertools` library was used to find the best thresholds for each class. These thresholds determine when a class is considered positive.

#### Impact of Optimized Thresholds:
- Without class-specific thresholds, metrics (especially **recall**) were approximately **1-1.5% lower**.
- **F1-score** was used to optimize these thresholds. The formula for F1-score is:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

### 8. Model Performance on Validation Data
Using all the techniques mentioned above, the model achieved the following metrics on the validation dataset:
- **Accuracy**: `0.95`
- **Precision**: `0.97`
- **Recall**: `0.96`
