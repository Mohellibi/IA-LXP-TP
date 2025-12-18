import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sqlalchemy import create_engine

# --- Configuration MongoDB ---
MONGO_URI = "mongodb+srv://thomascoutarel_db_user:MxcAu90w6P0ldNzv@googlereviews.jhi4anr.mongodb.net/?appName=googlereviews"
DATABASE_NAME = "googlereviews"
COLLECTION_NAME = "google_reviews"

# --- Configuration MySQL Workbench ---
MYSQL_USER = "root"
MYSQL_PASSWORD = "!4$b7WsJwM5&eznjCS#C"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "BIGDATA3"
MYSQL_TABLE = "google_reviews"

# --- 1. Récupération des données depuis MongoDB ---
try:
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    mongo_data = list(collection.find({}))
    client.close()
    print(f"{len(mongo_data)} documents récupérés depuis MongoDB.")

    # --- 2. Chargement dans Pandas et nettoyage ---
    mongo_df = pd.DataFrame(mongo_data)
    mongo_df["Provenance"] = "Google Play"

    cols_to_drop = ["userName", "userImage", "thumbsUpCount", "replyContent",
                    "repliedAt", "reviewId", "reviewCreatedVersion", "_id"]
    mongo_df_clean = mongo_df.drop(columns=cols_to_drop, errors="ignore")
    mongo_df_clean = mongo_df_clean.drop_duplicates()
    mongo_df_clean.rename(columns={'at': 'date'}, inplace=True)

    print("Nettoyage terminé. Aperçu :")
    print(mongo_df_clean.head())

except Exception as e:
    print(f" Erreur MongoDB : {e}")
    exit(1)

# --- 3. Connexion MySQL et insertion ---
try:
    # Créer l'engine SQLAlchemy
    engine = create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

    # Insérer le DataFrame dans MySQL (remplace si table existante)
    mongo_df_clean.to_sql(name=MYSQL_TABLE, con=engine, if_exists='replace', index=False)
    print(f" Données insérées dans MySQL table '{MYSQL_TABLE}' avec succès.")

except Exception as e:
    print(f"Erreur MySQL : {e}")
