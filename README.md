# 🧠 Multi-Label Emotion Classifier using BERT

This project is a multi-label emotion classification system built with **BERT** and **PyTorch**. It can detect multiple emotions (e.g., anger, joy, fear) in a single piece of text. The model is trained and evaluated using standard NLP preprocessing, Hugging Face Transformers, and custom metrics.

## 🚀 Features

- BERT-based architecture for contextual language understanding
- Supports multiple simultaneous emotion labels per input
- Clean text preprocessing pipeline
- Evaluation metrics: Hamming Loss, Micro-F1, Macro-F1
- Easy prediction on custom input text

---

## 🧾 Emotion Labels

The classifier supports the following emotion labels:

- Anger
- Fear
- Joy
- Sadness
- Surprise

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/i-abdul-raheem/nlp-milestone-3.git
   cd nlp-milestone-3
   ```

2.	Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```