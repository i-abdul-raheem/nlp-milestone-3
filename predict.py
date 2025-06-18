import torch
from transformers import BertTokenizer
import pandas as pd
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text
import numpy as np

# Load model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BERTMultiLabelClassifier()
model.load_state_dict(torch.load("model.pth", map_location=device))  # Adjust path if needed
model.to(device)
model.eval()

# Define label list
LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']  # Update if different

def predict(texts):
    cleaned = [clean_text(t) for t in texts]
    encodings = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = (outputs > 0.5).int().cpu().numpy()

    results = []
    for i, pred in enumerate(predictions):
        labels = [LABELS[j] for j, val in enumerate(pred) if val == 1]
        results.append({"text": texts[i], "emotions": labels})
    return results

if __name__ == "__main__":
    sample_texts = [
        "I'm feeling so happy and excited today!",
        "This is the worst day of my life.",
        "I don’t know what will happen, I’m scared."
    ]
    predictions = predict(sample_texts)
    for p in predictions:
        print(f"Text: {p['text']}\nPredicted Emotions: {p['emotions']}\n")