# RestaurantFindr — FastAPI + React

Versión alternativa de la app con FastAPI + React. Lee los mismos datos
generados por el pipeline principal (clean.csv, cluster_assignments.parquet,
results.csv).

## Requisitos

- Pipeline ejecutado (clean.csv + cluster_assignments.parquet generados)
- Node.js 18+
- Python con las mismas dependencias del proyecto (uv sync)

## Para la pestaña En vivo

El tab Live requiere que el stream esté activo:

**Terminal 0 — Producer + Scorer:**
```bash
python tasks/producer.py
```
```bash
python streaming_score.py
```

## Arrancar

**Terminal 1 — Backend:**
```bash
cd webapp/backend
source ../../.venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd webapp/frontend
npm install
npm run dev
```

App en http://localhost:5173
