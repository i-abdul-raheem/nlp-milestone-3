from sklearn.metrics import f1_score, hamming_loss

def calculate_metrics(y_true, y_pred):
    return {
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Micro F1': f1_score(y_true, y_pred, average='micro', zero_division=0)
    }