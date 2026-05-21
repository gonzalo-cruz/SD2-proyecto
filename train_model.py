# Entrena un modelo KMeans sobre el dataset de restaurantes usando Spark ML.
# El proceso tiene dos fases:
#   1. Selección de K: evalúa varios valores de K sobre una muestra del 10%
#      usando silhouette score y elige el mejor.
#   2. Entrenamiento final: entrena con el K elegido sobre el dataset completo
#      y guarda el modelo + las asignaciones de cluster por restaurante.
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

BASE_DIR = Path(__file__).parent
PREPROCESSED = BASE_DIR / "data" / "processed" / "preprocessed.csv"
TYPE_DICT = BASE_DIR / "data" / "processed" / "type_dict_encoding.json"
MODELS_DIR = BASE_DIR / "models"

# Valores de K a comparar en la fase de selección
K_RANGE = [10, 20, 30, 40, 50, 60, 70, 80]
SAMPLE_FRACTION = 0.10   # fracción del dataset para la selección de K
SEED = 42
MAX_ITER = 20


def select_k(sdf_feat_sample):
    # Probamos cada K y calculamos su silhouette score
    # Silhouette mide qué tan bien separados están los clusters (1 = perfecto, -1 = malo)
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    results = []
    for k in K_RANGE:
        print(f"  K={k:3d} ... ", end="", flush=True)
        m     = KMeans(k=k, seed=SEED, maxIter=MAX_ITER,
                       featuresCol="features", predictionCol="cluster",
                       distanceMeasure="cosine").fit(sdf_feat_sample)
        wssse = m.summary.trainingCost
        sil   = evaluator.evaluate(m.transform(sdf_feat_sample))
        results.append({"k": k, "wssse": wssse, "silhouette": sil})
        print(f"WSSSE={wssse:>14,.0f}  Silhouette={sil:.4f}")

    # Elegimos el K con mayor silhouette (mejor separación entre clusters)
    best   = max(results, key=lambda r: r["silhouette"])
    best_k = best["k"]

    print(f"\n  K óptimo (mayor silhouette): {best_k}  "
          f"(score={best['silhouette']:.4f})")

    _plot_k_selection(results, best_k)
    return best_k


def _plot_k_selection(results, best_k):
    # Generamos dos gráficas: método del codo (WSSSE) y silhouette score por K
    try:
        import matplotlib.pyplot as plt

        k_arr     = [r["k"]         for r in results]
        wssse_arr = [r["wssse"]     for r in results]
        sil_arr   = [r["silhouette"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Selección de K — KMeans", fontsize=13)

        ax1.plot(k_arr, wssse_arr, "b-o", linewidth=2, markersize=6)
        ax1.set_xlabel("K")
        ax1.set_ylabel("WSSSE")
        ax1.set_title("Método del Codo")
        ax1.grid(True, alpha=0.3)

        ax2.plot(k_arr, sil_arr, "g-o", linewidth=2, markersize=6)
        ax2.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
                    label=f"K óptimo = {best_k}")
        ax2.set_xlabel("K")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Puntuación de Silueta")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = MODELS_DIR / "k_selection.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Gráfica guardada → {out}")
    except ImportError:
        pass


def train(spark, feature_cols, pdf, feat_matrix, k):
    import shutil
    tmp = MODELS_DIR / "_tmp_features"
    n   = len(pdf)

    # Spark 4.x necesita leer parquet de un directorio, no de un fichero suelto
    print("Escribiendo parquet temporal para Spark...")
    tmp.mkdir(exist_ok=True)
    pdf.to_parquet(tmp / "data.parquet", index=False)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="keep")
    sdf_feat  = assembler.transform(spark.read.parquet(str(tmp))).select("row_id", "features")

    print(f"Entrenando KMeans (k={k}, maxIter={MAX_ITER})...")
    model = KMeans(k=k, seed=SEED, maxIter=MAX_ITER,
                   featuresCol="features", predictionCol="cluster",
                   distanceMeasure="cosine").fit(sdf_feat)
    print(f"  WSSSE: {model.summary.trainingCost:,.2f}")

    # Obtenemos el cluster asignado a cada restaurante
    print("Obteniendo asignaciones de cluster...")
    pdf_clusters = (model.transform(sdf_feat)
                    .select("row_id", "cluster")
                    .toPandas()
                    .sort_values("row_id")
                    .reset_index(drop=True))

    # Calculamos la distancia de cada restaurante a su centroide con numpy
    # (más rápido que hacerlo en Spark para este caso)
    print("Calculando distancias a centroides...")
    centers     = np.array(model.clusterCenters(), dtype=np.float32)
    cluster_ids = pdf_clusters["cluster"].values.astype(int)
    dists       = np.linalg.norm(feat_matrix - centers[cluster_ids], axis=1).astype(np.float32)

    # Guardamos las asignaciones en parquet para que la app las lea rápido
    pd.DataFrame({
        "row_id":           pdf_clusters["row_id"].astype(np.int32),
        "cluster":          pdf_clusters["cluster"].astype(np.int16),
        "dist_to_centroid": dists,
    }).to_parquet(MODELS_DIR / "cluster_assignments.parquet", index=False)

    with open(MODELS_DIR / "cluster_centers.json", "w") as f:
        json.dump(centers.tolist(), f)

    # Guardamos los nombres de features para que streaming_score.py use exactamente las mismas
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # overwrite() es necesario si ya existe el modelo de una ejecución anterior
    model.write().overwrite().save(str(MODELS_DIR / "kmeans_spark"))
    shutil.rmtree(tmp)

    sizes = pdf_clusters["cluster"].value_counts()
    print(f"\nArtefactos guardados en models/")
    print(f"Clusters: min={sizes.min():,}  max={sizes.max():,}  "
          f"media={sizes.mean():.0f} restaurantes/cluster")

    return model


def sanitize_col(name: str) -> str:
    # Spark interpreta '.' en nombres de columna como acceso a struct field.
    # Los OHE generan nombres como 'avg_rating__1.0' → los sustituimos por '_'
    return name.replace(".", "_")


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # Leemos el diccionario de tipos para saber qué columnas son features del modelo
    with open(TYPE_DICT) as f:
        type_dict = json.load(f)
    feature_cols      = [c for c, t in type_dict.items() if t in ("numeric", "ohe")]
    safe_feature_cols = [sanitize_col(c) for c in feature_cols]
    print(f"Feature columns: {len(feature_cols)} (numeric + ohe)")

    print("Leyendo preprocessed.csv...")
    pdf = pd.read_csv(PREPROCESSED, usecols=feature_cols, dtype=np.float32)
    pdf.fillna(0.0, inplace=True)
    n_rows = len(pdf)
    # row_id = posición en el CSV, sirve como clave para cruzar con clean.csv en la app
    pdf.insert(0, "row_id", np.arange(n_rows, dtype=np.int32))
    pdf.rename(columns={c: sanitize_col(c) for c in feature_cols}, inplace=True)
    feat_matrix = pdf[safe_feature_cols].values
    print(f"  {n_rows:,} filas × {len(safe_feature_cols)} features")

    spark = (SparkSession.builder
             .appName("KMeansTraining")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Si ya existe best_k.json nos saltamos la fase de selección de K
    # para no repetir un proceso que tarda varios minutos
    best_k_path = MODELS_DIR / "best_k.json"
    if best_k_path.exists():
        k = json.load(open(best_k_path))["k"]
        print(f"\nK cacheado: {k}  (elimina models/best_k.json para repetir selección)")
    else:
        n_sample = int(n_rows * SAMPLE_FRACTION)
        print(f"\nSelección de K sobre {SAMPLE_FRACTION*100:.0f}% de los datos ({n_sample:,} filas)...")

        pdf_sample = pdf.sample(n=n_sample, random_state=SEED).reset_index(drop=True)

        # Con 100K filas createDataFrame es suficientemente rápido sin fichero intermedio
        assembler       = VectorAssembler(inputCols=safe_feature_cols, outputCol="features",
                                          handleInvalid="keep")
        sdf_feat_sample = assembler.transform(
            spark.createDataFrame(pdf_sample)
        ).select("features").cache()

        k = select_k(sdf_feat_sample)

        sdf_feat_sample.unpersist()
        json.dump({"k": k}, open(best_k_path, "w"))

    print(f"\nEntrenando modelo final con K={k}...")
    train(spark, safe_feature_cols, pdf, feat_matrix, k)

    spark.stop()
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
