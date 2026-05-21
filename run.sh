#!/bin/bash

URL_FRONTEND="http://localhost:5173"
mkdir -p logs

echo "==== 0. Preparando Entorno ===="
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    \. "$NVM_DIR/nvm.sh"
    nvm use 22 > /dev/null 2>&1
fi

echo ""
echo "==== 1. Iniciando Infraestructura (Docker) ===="
docker compose up -d
echo "Esperando 12 segundos para que Kafka inicialice puertos y broker..."
sleep 12

echo "==== 2. Autocreando Topics en Kafka ===="
KAFKA_CONTAINER=$(docker ps --filter "name=local-kafka" --format "{{.Names}}" | head -n 1)
[ -z "$KAFKA_CONTAINER" ] && KAFKA_CONTAINER=$(docker ps --filter "name=kafka" --format "{{.Names}}" | head -n 1)

if [ -n "$KAFKA_CONTAINER" ]; then
    for topic in restaurants_schema restaurants restaurants_priority; do
        docker exec "$KAFKA_CONTAINER" kafka-topics --create --topic "$topic" --bootstrap-server localhost:9092 --if-not-exists > /dev/null 2>&1
    done
    echo "Topics base preparados con éxito."
fi

echo ""
echo "==== 3. Limpiando resultados de streaming anteriores ===="
rm -f data/streaming/results.csv data/streaming/filter_results.csv data/streaming/seen_row_ids.json
echo "Archivos de streaming limpiados."

echo "Dando margen de 6s para la publicación de esquemas..."
sleep 6

echo "==== 4. Iniciando Consumidores de Streaming (Spark) ===="
echo "Ejecutando streaming_score.py (Clasificación KMeans ML)..."
uv run streaming_score.py > logs/streaming_score.log 2>&1 &
SCORE_PID=$!

echo "Ejecutando streaming_filter.py (Filtro Rápido en Tiempo Real)..."
uv run streaming_filter.py > logs/streaming_filter.log 2>&1 &
FILTER_PID=$!

echo ""
echo "==== 5. Iniciando Backend y Frontend ===="

# Entramos, ejecutamos en segundo plano y salimos (de forma segura)
cd webapp/backend || exit 1
uv run uvicorn main:app --reload > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

echo "Esperando a que el backend cargue el dataset en memoria (puerto 8000)..."
# Bucle que espera a que el puerto 8000 responda
while ! (echo > /dev/tcp/127.0.0.1/8000) >/dev/null 2>&1; do 
    sleep 1
done

echo "Backend listo y escuchando en el puerto 8000."
echo "Arrancando Servidor de Desarrollo del Frontend (NPM)..."

# Entramos al frontend de forma segura
cd webapp/frontend || exit 1
npm run dev > ../../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

sleep 1
echo ""
echo "==== 6. Abriendo la aplicación en el navegador ===="
if command -v xdg-open > /dev/null; then xdg-open "$URL_FRONTEND"
elif command -v open > /dev/null; then open "$URL_FRONTEND"
elif command -v start > /dev/null; then start "$URL_FRONTEND"
fi

cleanup() {
    echo -e "\n\n==== Deteniendo procesos en segundo plano... ===="
    kill $SCORE_PID $FILTER_PID $PRODUCER_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null
    
    echo "==== Limpiando puertos residuales ===="
    if command -v lsof > /dev/null; then
        lsof -ti:8000 | xargs -r kill -9 2>/dev/null
        lsof -ti:5173 | xargs -r kill -9 2>/dev/null
    elif command -v fuser > /dev/null; then
        fuser -k -9 8000/tcp > /dev/null 2>&1
        fuser -k -9 5173/tcp > /dev/null 2>&1
    fi

    echo "==== Cazando procesos zombis de Spark/Python ===="
    pkill -f "streaming_score.py" 2>/dev/null || true
    pkill -f "streaming_filter.py" 2>/dev/null || true
    pkill -f "producer.py" 2>/dev/null || true

    echo "==== Apagando Infraestructura (Docker) ===="
    docker compose down
    echo "==== ¡Todo detenido de forma limpia y segura! ===="
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "=========================================================="
echo " ¡Todos los componentes se están ejecutando con éxito!"
echo "=========================================================="
echo " Presiona [ENTER] aquí en cualquier momento para detener todo..."
read -r
cleanup