import torch
from transformers import BertTokenizer
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text
import pandas as pd
import os
import argparse

# Constants
LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTMultiLabelClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")

    texts = df["text"].fillna("").tolist()
    cleaned_texts = [clean_text(text) for text in texts]
    
    encodings = tokenizer(cleaned_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)

    all_predictions = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = (outputs > 0.5).int().cpu().numpy()

        for pred in predictions:
            all_predictions.append(pred.tolist())

    return all_predictions

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Input CSV file path to analyze emotions")
    
    args = parser.parse_args()
    if args.csv_path:
        if not os.path.isfile(args.csv_path):
            raise FileNotFoundError(f"CSV file '{args.csv_path}' does not exist.")
        
        predictions = predict(args.csv_path)
        print(predictions)
        
        formatted_predictions = []
        for pred in predictions:
            emotions = [LABELS[i] for i, val in enumerate(pred) if val == 1]
            formatted_predictions.append(emotions)
        
        print(formatted_predictions)
    else:
        print("Please provide a valid CSV file path using --csv argument.")
        exit(1)

# Example usage:
# python main.py --csv data/track-a.csv
# This will read the CSV file, clean the text, and output the predicted emotions.
# Ensure the CSV file has a 'text' column with the text data to analyze.
# The output will be a list of dictionaries with the detected emotions for each text entry.
# Make sure to have the necessary libraries installed and the model trained before running this script.