import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text, encode_labels
from utils.evaluation import calculate_metrics
from models import BERTMultiLabelClassifier
import argparse

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        Args:
            texts (list): List of text samples.
            labels (list): List of labels corresponding to the texts.
            tokenizer (BertTokenizer): Tokenizer for encoding the texts.
            max_len (int): Maximum length of the tokenized sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """ Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
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

def train(args):
    """ Train the BERT model on the emotion dataset.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Load and preprocess dataset
    if not args.dataset:
        raise ValueError("Dataset path must be provided.")

    if not pd.io.common.file_exists(args.dataset):
        raise FileNotFoundError(f"Dataset file '{args.dataset}' does not exist.")

    print(f"Loading dataset from {args.dataset}")
    if not args.dataset.endswith('.csv'):
        raise ValueError("Dataset must be a CSV file.")

    # Read dataset
    print("Reading dataset...")
    try:
        df = pd.read_csv(args.dataset)
    except Exception as e:
        raise ValueError(f"Error reading dataset: {e}")

    if 'text' not in df.columns or 'labels' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'labels' columns.")

    print("Cleaning text data...")
    df['text'] = df['text'].apply(clean_text)

    print("Encoding labels...")
    labels = encode_labels(df)

    if len(labels) == 0:
        raise ValueError("No valid labels found in the dataset.")

    print("Splitting dataset into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'], labels, test_size=args.test_size, random_state=args.random_state
    )

    print("Tokenizing text data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = EmotionDataset(X_train.tolist(), y_train, tokenizer)
    val_dataset = EmotionDataset(X_val.tolist(), y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Initialize model
    print("Initializing model...")
    print(f"Using device: {device}")
    model = BERTMultiLabelClassifier().to(device)
    model.device = device
    model.train()
    
    print("Starting training...")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    for epoch in range(args.epoch):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels)

    print(calculate_metrics(all_labels, all_preds))
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    # Argument parser for command line arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/track-a.csv", help="Path to CSV dataset")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test size for validation split")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to use for training")
    parser.add_argument("--epoch", type=int, default=7, help="Number of training epochs")

    args = parser.parse_args()
    train(args)
