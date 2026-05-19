# Funciones para convertir el DataFrame de Polars en tablas de pandas
# listas para mostrar en Streamlit, decodificando los valores numéricos
# a sus etiquetas originales (país, ciudad, precio, etc.)
import numpy as np
import pandas as pd

from config import DISPLAY_LIMIT


def _decode(values, dmap):
    # Convierte una columna de códigos numéricos a sus nombres originales
    return [dmap.get(int(v), str(v)) if v is not None else "—" for v in values]


def build_display_df(subset, decode, limit=DISPLAY_LIMIT):
    # Tomamos los mejor valorados para no renderizar millones de filas
    top = subset.sort("avg_rating", descending=True).head(limit)
    return pd.DataFrame({
        "Nombre":   _decode(top["restaurant_name"], decode["restaurant_name"]),
        "País":     _decode(top["country"],         decode["country"]),
        "Ciudad":   _decode(top["city"],             decode["city"]),
        "Rating":   _decode(top["avg_rating"],       decode["avg_rating"]),
        "Precio":   _decode(top["price_level"],      decode["price_level"]),
        "Veg.":     ["✓" if v == 1 else "" for v in top["vegetarian_friendly"]],
        "Vegano":   ["✓" if v == 1 else "" for v in top["vegan_options"]],
        "Sin gl.":  ["✓" if v == 1 else "" for v in top["gluten_free"]],
        # Columnas internas (prefijo _) usadas para la lógica de recomendaciones,
        # no se muestran directamente en la tabla
        "_row_id":  top["row_id"].to_list(),
        "_cluster": top["cluster"].to_list(),
        "_dist":    top["dist_to_centroid"].to_list(),
    })


def build_rec_df(recs, decode):
    display  = build_display_df(recs, decode, limit=10)
    dists    = np.array(display["_dist"].tolist(), dtype=np.float32)
    max_dist = dists.max() + 1e-8
    # Convertimos la distancia al centroide en un porcentaje de similitud (0-100)
    # donde 100 = idéntico al centroide y 0 = el más alejado del grupo
    display["Similitud"] = [int(round((1 - d / max_dist) * 100)) for d in dists]
    return display
