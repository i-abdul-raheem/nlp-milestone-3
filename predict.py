import torch
from transformers import BertTokenizer
import pandas as pd
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text
import argparse

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BERTMultiLabelClassifier()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze emotions")
    parser.add_argument("--model", type=str, default="model.pth", help="Path to the trained model file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run inference on")
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    predictions = predict([args.text])
    for p in predictions:
        print(f"Text: {p['text']}\nPredicted Emotions: {p['emotions']}\n")