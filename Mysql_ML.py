import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords
import mysql.connector

# --- 1. Récupération des données depuis MySQL ---
MYSQL_CONFIG = {
    'user': 'root',
    'password': '!4$b7WsJwM5&eznjCS#C',
    'host': 'localhost',
    'database': 'BIGDATA3',
    'port': 3306
}

conn = mysql.connector.connect(**MYSQL_CONFIG)
df = pd.read_sql("SELECT * FROM google_reviews", conn)
conn.close()
print(f"{len(df)} lignes récupérées depuis MySQL")

# --- 2. Prétraitement minimal des commentaires ---
df['content'] = df['content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df = df.dropna(subset=['content', 'score'])
df = df.reset_index(drop=True)

# --- 3. Diviser en train/test ---
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['score'], test_size=0.2, random_state=42)

# --- 4. Calcul des poids de classe pour LogisticRegression ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# --- 5. TF-IDF vectorization ---
nltk.download('stopwords')
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords.words('english'))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- 6. Modèle 1 : Logistic Regression ---
lr_model = LogisticRegression(max_iter=10000, class_weight=class_weight_dict)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
mse_lr = mean_squared_error(y_test, y_pred_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression - MSE : {mse_lr:.4f}, Accuracy : {acc_lr:.4f}")

# --- 7. Modèle 2 : Random Forest Classifier optimisé ---
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_features='sqrt',
    max_depth=20
)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest - MSE : {mse_rf:.4f}, Accuracy : {acc_rf:.4f}")

# --- 8. Préférences / importance des features pour Logistic Regression ---
feature_names = vectorizer.get_feature_names_out()
coef_lr = lr_model.coef_[0]

top_pos_lr = sorted(zip(feature_names, coef_lr), key=lambda x: x[1], reverse=True)[:10]
top_neg_lr = sorted(zip(feature_names, coef_lr), key=lambda x: x[1])[:10]

