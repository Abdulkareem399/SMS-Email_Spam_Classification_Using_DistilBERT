 SpamShield AI — SMS/Email Spam Classification

An end-to-end NLP project that fine-tunes a DistilBERT transformer model to classify text messages and emails as Spam or Ham (Safe), served through a clean, interactive Flask web application.

📌 Overview

This project trains a transformer-based binary text classifier to detect spam messages with high accuracy, then exposes the model through a lightweight web interface where users can paste any message and get an instant prediction with a confidence score.


✨ Features


🤖 Fine-tuned DistilBERT (distilbert-base-uncased) for binary text classification
🧹 Full NLP preprocessing pipeline (lowercasing, URL/punctuation removal, stopword filtering)
📊 Model evaluation with Accuracy, Precision, Recall, and F1-score
🌐 Flask backend serving real-time predictions via a /predict API endpoint
🎨 Responsive, modern frontend with live confidence meter and sample messages
⚡ Fast inference suitable for local or lightweight production use



🧠 Model Details

PropertyValueBase modeldistilbert-base-uncasedTaskSequence Classification (binary)Labelsham (0), spam (1)FrameworkHugging Face transformers (Trainer API)Max sequence length128 tokensEpochs2Learning rate2e-5Evaluation Accuracy 99.50%Validation F1-Score 98.15%


🛠️ Tech Stack


ML/NLP: PyTorch, Hugging Face Transformers, Datasets, scikit-learn, NLTK
Backend: Flask
Frontend: HTML, CSS, JavaScript (Font Awesome icons)
Data: SMS Spam Collection dataset (Category, Message columns)


🚀 Getting Started

1. Clone the repository

bashgit clone https://github.com/<your-username>/SMS_Spam_Classification.git
cd SMS_Spam_Classification

2. Create a virtual environment & install dependencies

bashpython -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
pip install torch transformers datasets scikit-learn nltk flask pandas numpy

3. Train the model (optional — if results_final/ isn't already present)

Open and run all cells in main.ipynb. This will:


Clean and preprocess the dataset
Fine-tune DistilBERT on the spam/ham labels
Save the final model and tokenizer to ./results/results_final


4. Run the web app

bashpython app.py

Then open your browser at:

http://127.0.0.1:5000


🔍 How It Works


User enters a message in the web UI.
The message is sent to the Flask /predict endpoint.
The fine-tuned DistilBERT tokenizer encodes the text.
The model outputs class probabilities via Softmax.
The predicted label (spam/ham) and confidence score are returned and displayed live.



📈 Example Predictions

MessagePrediction"Congratulations! You won a FREE iPhone 16..."🔴 Spam"Hey, are we still meeting for lunch at 1 PM?"🟢 Ham (Safe)


📜 License

This project is open-sourced under the MIT License.


🙋 Author

Built by Abdul kareem as a hands-on NLP + Deep Learning project applying transformer-based text classification to a real-world spam detection problem.
