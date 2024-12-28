import torch
from torch.utils.data import TensorDataset, DataLoader


def predict_toxicity(input_text, model, tokenizer, device='cpu'): 
    # Tokenize input text
    user_encodings = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=128,  # Ensure compatibility with the model
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
    
    return predicted_labels[0]  # Return as a simple list