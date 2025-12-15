from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd

from google_play_scraper import reviews
from app_store_scraper import AppStore

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

# --- PARAMÈTRES ---
countries = ["us", "fr", "de"]
app_id_google = "jp.pokemon.pokemontcgp"

MONGO_URI = "mongodb+srv://thomascoutarel_db_user:MxcAu90w6P0ldNzv@googlereviews.jhi4anr.mongodb.net/?appName=googlereviews"
DATABASE_NAME = "googlereviews"
COLLECTION_NAME = "google_reviews"

MAX_INSERT = 20000  # nombre maximal de documents à insérer
REVIEWS_PER_COUNTRY = MAX_INSERT // len(countries)  # réparti sur les pays
BATCH_SIZE = 1000  # MongoDB Atlas batch

# --- SCRAPING GOOGLE PLAY ---
all_google_reviews = []

for country in countries:
    print(f"Scraping Google Play Store pour : {country}")

    result, _ = reviews(
        app_id_google,
        lang="en",
        country=country,
        count=REVIEWS_PER_COUNTRY
    )

    for review in result:
        review["Pays"] = country
        all_google_reviews.append(review)

    print(f"{len(result)} avis récupérés pour {country}")

print("Scraping terminé.")

# --- DATAFRAME ---
google_df = pd.DataFrame(all_google_reviews)
print(f"Total avis récupérés : {len(google_df)}")
google_df = google_df.head(MAX_INSERT)
print(f"Limité à {len(google_df)} avis pour MongoDB")

# --- EXPORT MONGODB (INSERT PAR BATCH) ---
try:
    client = MongoClient(
        MONGO_URI,
        server_api=ServerApi("1"),
        socketTimeoutMS=60000,
        connectTimeoutMS=20000
    )

    client.admin.command("ping")
    print("Connexion MongoDB réussie")

    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    data_to_insert = google_df.to_dict("records")
    total_inserted = 0

    for i in range(0, len(data_to_insert), BATCH_SIZE):
        batch = data_to_insert[i:i + BATCH_SIZE]
        result = collection.insert_many(batch)
        total_inserted += len(result.inserted_ids)
        print(f"{total_inserted}/{len(data_to_insert)} documents insérés")

    client.close()
    print("Insertion MongoDB terminée")

except Exception as e:
    print(f"Erreur MongoDB : {e}")
