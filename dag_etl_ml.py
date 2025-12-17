from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'etl_ml_pipeline',
    default_args=default_args,
    description='ETL pipeline: Scrape reviews -> Clean to MySQL -> Run ML',
    schedule_interval=timedelta(days=1),  # Run daily; adjust as needed
    catchup=False,
)

# Path to Python executable (update if using a venv)
python_path = 'python'  # e.g., '/path/to/.venv/bin/python'

# Task 1: Run scraping script to MongoDB
scrape_task = BashOperator(
    task_id='scrape_to_mongodb',
    bash_command=f'{python_path} Scrapp_google_to_mongodb.py',
    dag=dag,
)

# Task 2: Run cleaning and transfer to MySQL
clean_task = BashOperator(
    task_id='clean_mongodb_to_mysql',
    bash_command=f'{python_path} Cleaning_Mongodb_toMysql.py',
    dag=dag,
)

# Task 3: Run ML analysis on MySQL data
ml_task = BashOperator(
    task_id='run_ml_analysis',
    bash_command=f'{python_path} Mysql_ML.py',
    dag=dag,
)

# Set task dependencies: Scrape -> Clean -> ML
scrape_task >> clean_task >> ml_task
