import numpy as np

def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Hamming Loss
    hamming = np.sum(y_true != y_pred) / (y_true.shape[0] * y_true.shape[1])

    # Macro F1
    macro_f1 = 0
    for i in range(y_true.shape[1]):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        macro_f1 += f1
    macro_f1 /= y_true.shape[1]

    # Micro F1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'Hamming Loss': float(hamming),
        'Macro F1': float(macro_f1),
        'Micro F1': float(micro_f1)
    }
    
if __name__ == "__main__":
    y_true = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    y_pred = [[1, 0, 0], [0, 1, 1], [1, 0, 0]]
    
    metrics = calculate_metrics(y_true, y_pred)
    print(metrics)
