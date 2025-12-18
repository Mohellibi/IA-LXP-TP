import subprocess
import sys
import os

print(f"Current working directory: {os.getcwd()}")
print("Running Mysql_ML.py to train the model and save the vectorizer...")
result = subprocess.run([sys.executable, "Mysql_ML.py"])
if result.returncode != 0:
    print("Error in Mysql_ML.py")
    sys.exit(1)
else:
    print("Mysql_ML.py completed successfully.")

print("Checking if fitted_vectorizer.pkl exists after Mysql_ML.py...")
if os.path.exists("fitted_vectorizer.pkl"):
    print("fitted_vectorizer.pkl found.")
else:
    print("fitted_vectorizer.pkl NOT found!")

print("Starting the API server...")
subprocess.run([sys.executable, "api_mlflow_rf.py"])
