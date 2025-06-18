import torch
import pandas as pd
from transformers import BertTokenizer
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text

# Config
LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
MODEL_PATH = 'model.pth'
CSV_PATH = 'data/track-a.csv'
NUM_SAMPLES = 50

# Load model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BERTMultiLabelClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load and sample data
df = pd.read_csv(CSV_PATH)
df['text'] = df['text'].apply(clean_text)
samples = df.sample(n=NUM_SAMPLES, random_state=42)['text'].tolist()

# Predict function
def predict(texts):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = (outputs > 0.5).int().cpu().numpy()

    results = []
    for i, pred in enumerate(preds):
        emotions = [LABELS[j] for j, val in enumerate(pred) if val == 1]
        results.append((texts[i], emotions))
    return results

# Run prediction and print
results = predict(samples)
for text, emotions in results:
    print(f"Text: {text}\nPredicted Emotions: {emotions}\n")