
import sys
from pathlib import Path

from airflow.sdk import dag, task
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).parent.parent)

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
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from tasks.extract import extract as run
        run()

    @task()
    def clean():
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from tasks.clean import clean as run
        run()

    @task()
    def eda():
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from tasks.eda import eda as run
        run()

    @task()
    def preprocessing():
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from tasks.preprocessing import preprocessing as run
        run()

    @task()
    def load():
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from tasks.load import load as run
        run()

    extract() >> clean() >> eda() >> preprocessing() >> load()


tripadvisor_pipeline()
