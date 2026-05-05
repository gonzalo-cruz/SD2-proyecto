import sys
from pathlib import Path
from datetime import datetime

from airflow.sdk import dag, task

PROJECT_ROOT = str(Path(__file__).parent.parent)

# Inyectamos el PROJECT_ROOT en el path para todos los workers/tareas
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tasks.load import load

"""
Orden: extract → clean → eda → preprocessing → load
"""

@dag(
    dag_id="tripadvisor_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tripadvisor", "restaurants"],
)
def tripadvisor_pipeline():

    @task()
    def extract():
        from tasks.extract import extract as run
        run()

    @task()
    def clean():
        from tasks.clean import clean as run
        run()

    @task()
    def eda():
        from tasks.eda import eda as run
        run()

    @task()
    def preprocessing():
        from tasks.preprocessing import preprocessing as run
        run()

    # Asignamos las tareas de Python a variables
    t_extract = extract()
    t_clean = clean()
    t_eda = eda()
    t_preprocessing = preprocessing()
    t_load = load()

    # Definimos el flujo
    t_extract >> t_clean >> t_eda >> t_preprocessing >> t_load

tripadvisor_pipeline()