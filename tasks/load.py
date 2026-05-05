from pathlib import Path
from airflow.providers.standard.operators.bash import BashOperator

def load() -> BashOperator:
    """
    Construye y retorna el BashOperator para la tarea de carga (load).
    """
    # Obtiene la ruta (la carpeta tasks)
    current_dir = Path(__file__).parent.resolve()
    
    # Resuelve la ruta hacia producer.py
    producer_script = current_dir / "producer.py"

    # Retorna el operador
    return BashOperator(
        task_id="load",
        bash_command=f"set -e; python {producer_script}"
    )