from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# --- SMART PATH LOOKUP ---
# Check common directories where the notebook may have generated the 'results_final' folder
# --- SMART PATH LOOKUP ---
# This looks directly into your specific Windows directory layout
POSSIBLE_PATHS = [
    "./results/results_final",
    "results/results_final",
    "./results_final",
    "E:/DS all algorithms projects/SMS_Spam_Classification/results/results_final"
]

MODEL_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(os.path.join(path, "config.json")):
        MODEL_PATH = path
        break

if MODEL_PATH:
    try:
        # Explicitly loading from the verified directory path
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print(f"\n🚀 SUCCESS: Loaded fine-tuned local model weights from: '{MODEL_PATH}'\n")
    except Exception as e:
        print(f"\n❌ Error parsing folder weights at {MODEL_PATH}: {e}")
        MODEL_PATH = None

if not MODEL_PATH:
    print("\n⚠️ WARNING: Fine-tuned folder 'results/results_final' could not be found!")
    print("Loading base model checkpoint as a fallback...\n")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1}
    )

model.eval()

@app.route('/')
def home():
    # Renders the index.html template from the /templates directory
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    text = data['message']
    
    # Tokenize input text to match training specifications
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    
    # Model inference (disable gradient updates for faster production inference)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate confidence probabilities using Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_idx].item()
    
    # Map predictions back to original labels
    id2label = {0: "ham", 1: "spam"}
    label = id2label[pred_idx]
    
    # Return metrics payload back to front-end JavaScript as JSON
    return jsonify({
        'label': label,
        'confidence': f"{confidence * 100:.2f}%",
        'confidence_raw': confidence * 100
    })

if __name__ == '__main__':
    # use_reloader=False is critical to stop Windows Git Bash from crashing out of the session
    app.run(debug=True, port=5000, use_reloader=False)