import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text, encode_labels
from utils.evaluation import calculate_metrics
from models import BERTMultiLabelClassifier

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

def train():
    df = pd.read_csv('data/track-a.csv')
    df['text'] = df['text'].apply(clean_text)
    labels = encode_labels(df)

    X_train, X_val, y_train, y_val = train_test_split(df['text'], labels, test_size=0.3, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = EmotionDataset(X_train.tolist(), y_train, tokenizer)
    val_dataset = EmotionDataset(X_val.tolist(), y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    if torch.backends.mps.is_available():
        print("Using MPS backend for training.")
    else:   
        print("MPS backend not available, using CPU.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BERTMultiLabelClassifier().to(device)
    model.device = device
    model.train()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(7):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels)

    print(calculate_metrics(all_labels, all_preds))
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()