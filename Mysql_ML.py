# =========================
# Imports
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import stopwords

import mysql.connector
import mlflow
import mlflow.sklearn


# =========================
# MLflow config
# =========================
EXPERIMENT_NAME = "google_reviews_sentiment"
mlflow.set_experiment(EXPERIMENT_NAME)


# =========================
# MySQL config
# =========================
MYSQL_CONFIG = {
    "user": "root",
    "password": "!4$b7WsJwM5&eznjCS#C",
    "host": "localhost",
    "database": "BIGDATA3",
    "port": 3306
}


# =========================
# 1. Load data from MySQL
# =========================
conn = mysql.connector.connect(**MYSQL_CONFIG)
df = pd.read_sql("SELECT * FROM google_reviews", conn)
conn.close()

print(f"✅ {len(df)} lignes récupérées depuis MySQL")

# =========================
# 2. Preprocessing
# =========================
df = df.dropna(subset=["content", "score"])
df["content"] = (
    df["content"]
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
)

# =========================
# 3. Train / Test split
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["score"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# =========================
# 4. Class weights
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# =========================
# 5. TF-IDF Vectorizer
# =========================
nltk.download("stopwords")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=stopwords.words("english")
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================================
# 6. Logistic Regression
# ==========================================================
with mlflow.start_run(run_name="LogisticRegression"):

    lr_model = LogisticRegression(
        max_iter=10_000,
        class_weight=class_weight_dict
    )

    lr_model.fit(X_train_tfidf, y_train)
    y_pred = lr_model.predict(X_test_tfidf)

    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # ---- MLflow logging ----
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 10_000)
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("test_size", TEST_SIZE)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(lr_model, name="model")

    print(f"📌 LogisticRegression | accuracy={accuracy:.4f} | mse={mse:.4f}")


# ==========================================================
# 7. Random Forest
# ==========================================================
with mlflow.start_run(run_name="RandomForest"):

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    rf_model.fit(X_train_tfidf, y_train)
    y_pred = rf_model.predict(X_test_tfidf)

    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # ---- MLflow logging ----
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("max_features", "sqrt")
    mlflow.log_param("test_size", TEST_SIZE)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(rf_model, name="model")

    print(f"📌 RandomForest | accuracy={accuracy:.4f} | mse={mse:.4f}")


# =========================
# 8. Feature importance (LR)
# =========================
feature_names = vectorizer.get_feature_names_out()
coef = lr_model.coef_[0]

top_positive = sorted(
    zip(feature_names, coef),
    key=lambda x: x[1],
    reverse=True
)[:10]

top_negative = sorted(
    zip(feature_names, coef),
    key=lambda x: x[1]
)[:10]

print("\ Top mots positifs :", top_positive)
print(" Top mots négatifs :", top_negative)
