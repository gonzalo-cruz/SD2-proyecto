# Lógica de recomendación basada en los clusters de KMeans.
# La idea es sencilla: los restaurantes más similares son los del mismo cluster
# ordenados por su distancia al centroide (los más cercanos al centro del grupo
# son los más "representativos" y por tanto los mejores candidatos).
import polars as pl

from ui.filters import apply_filters


def get_recommendations(df, row_id, cluster, filters, encode, cuisine_idx, n=10):
    # Buscamos primero dentro del mismo cluster aplicando los filtros activos
    recs = df.filter((pl.col("cluster") == cluster) & (pl.col("row_id") != row_id))
    recs = apply_filters(recs, encode, cuisine_idx, filters)

    # Si los filtros son muy restrictivos y no hay suficientes resultados,
    # ampliamos la búsqueda a todo el dataset (manteniendo los filtros)
    if recs.height < n:
        recs = apply_filters(df.filter(pl.col("row_id") != row_id), encode, cuisine_idx, filters)

    # Ordenamos por distancia al centroide: menor distancia = más cercano al "tipo ideal"
    return recs.sort("dist_to_centroid").head(n)
