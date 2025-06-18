from flask import Flask, render_template_string, request, jsonify
import torch
from transformers import BertTokenizer
from models.bert_classifier import BERTMultiLabelClassifier
from utils.preprocessing import clean_text

app = Flask(__name__)

# Constants
LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTMultiLabelClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 2rem; background: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background: #fff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 100px; padding: 10px; font-size: 1rem; }
        button { margin-top: 1rem; padding: 10px 20px; font-size: 1rem; }
        .result, .loading { margin-top: 1rem; font-size: 1.2rem; }
    </style>
    <script>
        function handleSubmit(form) {
            const button = form.querySelector("button");
            const loading = document.getElementById("loading");
            button.disabled = true;
            loading.style.display = "block";
        }
    </script>
</head>
<body>
    <div class="container">
        <h2>Emotion Classifier</h2>
        <form method="post" onsubmit="handleSubmit(this)">
            <textarea name="text" placeholder="Enter your text here" required>{{ request.form.get('text', '') }}</textarea>
            <button type="submit">Predict</button>
        </form>
        <div id="loading" class="loading" style="display: none;">Predicting...</div>
        {% if prediction %}
            <div class="result">
                <strong>Predicted Emotion(s):</strong> {{ prediction }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def predict_emotions(text):
    cleaned = clean_text(text)
    encodings = tokenizer([cleaned], padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = (outputs > 0.5).int().cpu().numpy()[0]

    emotions = [LABELS[i] for i, val in enumerate(predictions) if val == 1]
    return emotions if emotions else ["neutral"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form.get("text", "")
        if text:
            prediction = ", ".join(predict_emotions(text))
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    emotions = predict_emotions(text)
    return jsonify({"text": text, "emotions": emotions})

if __name__ == "__main__":
    app.run(debug=True)