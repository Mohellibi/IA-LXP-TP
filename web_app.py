from flask import Flask, request, render_template
import torch
from transformers import pipeline
import re
import requests

app = Flask(__name__)

# Load pre-trained model
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        use_api = request.form.get("use_api") == "on"
        
        if not text:
            error = "Please enter a comment."
        else:
            try:
                if use_api:
                    # Call the API
                    response = requests.post("http://127.0.0.1:5001/predict", json={"text": text})
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get("prediction")
                        confidence = data.get("confidence", "N/A")
                    else:
                        error = f"API error: {response.text}"
                else:
                    # Local prediction
                    processed_text = preprocess_text(text)
                    result = sentiment_pipeline(processed_text)[0]
                    label = result['label']
                    prediction = int(label.split()[0])
                    confidence = round(result['score'], 2)
            except Exception as e:
                error = f"Prediction failed: {str(e)}"
    
    return render_template("index.html", prediction=prediction, confidence=confidence, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
