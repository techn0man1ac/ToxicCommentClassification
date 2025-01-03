import torch

# Function for model prediction
def predict_toxicity(input_text, model, tokenizer, device='cpu'):
    '''
    Implements prediction functionality of a transformer-based model created with Pytorch for text classification
    (in the specific case of the app - prediction of BERT-based model for toxicity of a given comment).

    Args:
        input_text (list[str]): list of texts (comments) to classify 
                               (with app interface a list with single comment is passed).
        model (torch.nn.Module): Pytorch model for text classification.
        tokenizer (transformers.PreTrainedTokenizer): tokenizer compatible with the model.
        device (str): name of device to run the process on.

    Returns:
        numpy.ndarray: list with binary values indicating presence of respective toxicity class.
    '''
    # Tokenize input text
    user_encodings = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Perform prediction
        outputs = model(
            input_ids=user_encodings['input_ids'],
            attention_mask=user_encodings['attention_mask']
        )
        logits = outputs.logits
        predictions = torch.sigmoid(logits)  # Apply sigmoid for multi-label classification

    # Threshold predictions and convert to binary labels
    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    
    return predicted_labels[0]
