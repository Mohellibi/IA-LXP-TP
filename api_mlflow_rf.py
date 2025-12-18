import nltk
import re
import pickle
from flask import Flask, request, jsonify

# =========================
# NLTK
# =========================
nltk.download("stopwords")

# =========================
# Global variables for lazy loading
# =========================
model = None
vectorizer = None


def load_model_and_vectorizer():
    global model, vectorizer
    if model is None or vectorizer is None:
        import mlflow
        import mlflow.sklearn
        import os

        print(f"Current working directory in API: {os.getcwd()}")
        print("Checking if fitted_vectorizer.pkl exists...")
        if os.path.exists("fitted_vectorizer.pkl"):
            print("fitted_vectorizer.pkl found, loading...")
        else:
            print("fitted_vectorizer.pkl NOT found!")

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Find the experiment
        experiments = mlflow.search_experiments(filter_string="name = 'google_reviews_sentiment'")
        if not experiments:
            raise ValueError("Experiment 'google_reviews_sentiment' not found. Please run Mysql_ML.py first.")
        
        experiment = experiments[0]
        print(f"Found experiment: {experiment.experiment_id}")
        
        # Find the latest RandomForest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="tags.mlflow.runName = 'RandomForest'", order_by=["start_time DESC"])
        if runs.empty:
            raise ValueError("No RandomForest run found")
        
        latest_run_id = runs.iloc[0].run_id
        print(f"Latest RandomForest run ID: {latest_run_id}")
        
        MODEL_URI = f"runs:/{latest_run_id}/model"
        print(f"Loading model from: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("Model loaded successfully.")
        
        # Load vectorizer from local file
        try:
            with open("fitted_vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            print("Vectorizer loaded successfully.")
        except FileNotFoundError:
            raise ValueError("fitted_vectorizer.pkl not found. Please run Mysql_ML.py first to generate it.")
        except Exception as e:
            raise ValueError(f"Error loading vectorizer: {e}")


# =========================
# Flask app
# =========================
app = Flask(__name__)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model_and_vectorizer()  # Load only if not already loaded
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    try:
        text = preprocess_text(data["text"])
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        
        return jsonify({
            "input": data["text"],
            "prediction": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
