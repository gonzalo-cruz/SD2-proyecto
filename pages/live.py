# Tab "En vivo": muestra los restaurantes que van llegando desde Kafka
# y han sido clasificados por Spark Structured Streaming (streaming_score.py).
# La tabla y las recomendaciones se actualizan automáticamente cada 5 segundos.
import pandas as pd
import polars as pl
import streamlit as st

from config import STREAM_CSV
from recommendations import get_recommendations
from ui.filters import apply_filters, build_filters
from ui.tables import build_rec_df


def render_live(df, decode, encode, cuisine_idx, top_cuisines, country_cities):
    st.markdown(
        '<div class="rf-hero">'
        '<div class="rf-logo" style="font-size:2rem">Stream <em>en vivo</em></div>'
        '<div class="rf-desc">'
        'Restaurantes llegando desde Kafka, clasificados por KMeans en tiempo real. '
        'Las recomendaciones y los filtros se aplican únicamente sobre los restaurantes '
        'que ya han sido scoreados. Actualización automática cada 5 segundos.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Si el fichero de resultados no existe todavía, el stream no ha arrancado
    if not STREAM_CSV.exists():
        st.info(
            "No hay datos aún. Arranca el pipeline en dos terminales:\n\n"
            "```\npython tasks/producer.py\n```\n"
            "```\npython streaming_score.py\n```"
        )
        return

    # Los filtros se renderizan FUERA del fragment para que el usuario pueda
    # interactuar con ellos sin que el auto-refresh los interrumpa.
    # El fragment lee los valores desde session_state en cada ciclo.
    build_filters(encode, top_cuisines=top_cuisines,
                  country_cities=country_cities, prefix="live_")
    st.markdown('<div class="rf-divider"></div>', unsafe_allow_html=True)

    @st.fragment(run_every="5s")
    def _feed():
        # Leemos el CSV que va creciendo conforme llegan batches de Spark
        raw = pd.read_csv(STREAM_CSV)
        if raw.empty:
            st.caption("Esperando el primer batch…")
            return

        # Cruzamos los row_ids que han llegado por stream con el DataFrame
        # completo para tener todas las columnas (país, ciudad, etc.)
        stream_row_ids = set(raw["row_id"].dropna().astype(int).tolist())
        stream_df      = df.filter(pl.col("row_id").is_in(stream_row_ids))

        # Leemos los filtros activos del session_state (seteados por los widgets de arriba)
        filters = dict(
            country    = st.session_state.get("live_country",  "Todos"),
            city       = st.session_state.get("live_city",     []),
            min_rating = st.session_state.get("live_rating",   1.0),
            price      = st.session_state.get("live_price",    "Todos"),
            cuisines   = st.session_state.get("live_cuisines", []),
            veg        = st.session_state.get("live_veg",      False),
            vegan      = st.session_state.get("live_vegan",    False),
            gf         = st.session_state.get("live_gf",       False),
        )

        filtered_stream = apply_filters(stream_df, encode, cuisine_idx, filters)
        total   = filtered_stream.height
        n_clust = raw["cluster"].nunique()

        # Estadísticas del stream actual
        st.markdown(
            f'<div class="rf-stats">'
            f'<div><span class="rf-stat-n">{len(raw):,}</span>'
            f'<span class="rf-stat-l">scoreados</span></div>'
            f'<div><span class="rf-stat-n">{total:,}</span>'
            f'<span class="rf-stat-l">con filtros</span></div>'
            f'<div><span class="rf-stat-n">{n_clust}</span>'
            f'<span class="rf-stat-l">clusters</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_table, col_chart = st.columns([2, 1], gap="large")

        with col_table:
            if total == 0:
                st.info("Ningún restaurante scoreado cumple los filtros aún.")
            else:
                # Enriquecemos el CSV del stream con los datos completos del restaurant
                filt_ids = filtered_stream["row_id"].to_list()
                enriched = (
                    df.filter(pl.col("row_id").is_in(filt_ids))
                      .select(["row_id", "country", "city", "avg_rating", "price_level"])
                      .to_pandas()
                )
                # Ordenamos por scored_at para ver los más recientes primero
                newest = (raw[raw["row_id"].isin(filt_ids)]
                            .sort_values("scored_at", ascending=False)
                            .reset_index(drop=True))
                merged = newest.merge(enriched, on="row_id", how="left")

                # Decodificamos los códigos numéricos a nombres legibles
                merged["País"]   = [decode["country"].get(int(v), "—") if pd.notna(v) else "—"
                                     for v in merged["country"]]
                merged["Ciudad"] = [decode["city"].get(int(v), "—") if pd.notna(v) else "—"
                                     for v in merged["city"]]
                merged["Rating"] = [decode["avg_rating"].get(int(v), "—") if pd.notna(v) else "—"
                                     for v in merged["avg_rating"]]
                merged["Precio"] = [decode["price_level"].get(int(v), "—") if pd.notna(v) else "—"
                                     for v in merged["price_level"]]

                show = merged[["name", "País", "Ciudad", "Rating", "Precio",
                                "cluster", "dist_to_centroid", "scored_at"]].head(100)
                show = show.rename(columns={
                    "name": "Restaurante", "cluster": "Cluster",
                    "dist_to_centroid": "Dist. centroide", "scored_at": "Recibido",
                })

                event = st.dataframe(
                    show, selection_mode="single-row", on_select="rerun",
                    use_container_width=True, hide_index=True, key="live_table",
                )

                # Guardamos la selección en session_state para que persista
                # entre ciclos de auto-refresh (el fragment se re-ejecuta cada 5s)
                sel = event.selection.rows
                if sel and sel[0] < len(merged):
                    st.session_state["live_rid"]     = int(merged.iloc[sel[0]]["row_id"])
                    st.session_state["live_cluster"] = int(merged.iloc[sel[0]]["cluster"])
                    st.session_state["live_name"]    = merged.iloc[sel[0]]["name"]

        with col_chart:
            st.caption("DISTRIBUCIÓN POR CLUSTER")
            cluster_counts = (
                raw["cluster"].value_counts()
                              .reset_index()
                              .rename(columns={"cluster": "Cluster", "count": "N"})
                              .sort_values("Cluster")
            )
            st.bar_chart(cluster_counts.set_index("Cluster")["N"])

        # Recomendaciones basadas en el restaurante seleccionado en el stream
        rid     = st.session_state.get("live_rid")
        cluster = st.session_state.get("live_cluster")
        name    = st.session_state.get("live_name", "")

        if rid is None:
            st.caption("Haz clic en un restaurante para ver similares dentro del stream.")
            return

        st.markdown('<div class="rf-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rf-rec-name">Similar a <em>{name}</em></div>'
            f'<div class="rf-rec-sub">Grupo {cluster} · solo restaurantes scoreados · con filtros</div>',
            unsafe_allow_html=True,
        )

        # Las recomendaciones solo se buscan entre los restaurantes ya scoreados
        recs = get_recommendations(filtered_stream, rid, cluster, filters, encode, cuisine_idx)
        if recs.height == 0:
            st.info("No hay similares en el stream con los filtros actuales. "
                    "Espera a que lleguen más restaurantes o ajusta los filtros.")
            return

        rec_df      = build_rec_df(recs, decode)
        rec_visible = [c for c in rec_df.columns if not c.startswith("_")]
        st.dataframe(
            rec_df[rec_visible],
            use_container_width=True, hide_index=True,
            column_config={
                "Similitud": st.column_config.ProgressColumn(
                    "Similitud", format="%d%%", min_value=0, max_value=100
                )
            },
        )

    _feed()
