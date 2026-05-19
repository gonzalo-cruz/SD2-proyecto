# Tab "Explorar": permite filtrar el millón de restaurantes y ver recomendaciones
# similares al hacer clic en uno. Las recomendaciones se calculan con KMeans.
import polars as pl
import streamlit as st

from config import DISPLAY_LIMIT
from recommendations import get_recommendations
from ui.filters import apply_filters, build_filters
from ui.tables import build_display_df, build_rec_df


def render_explore(df, decode, encode, cuisine_idx, top_cuisines, country_cities):
    # Cabecera de la página
    st.markdown(
        '<div class="rf-hero">'
        '<div class="rf-logo">Restaurant<em>Findr</em></div>'
        '<div class="rf-desc">'
        'Explora más de un millón de restaurantes europeos de TripAdvisor. '
        'Aplica filtros por país, ciudad, precio o tipo de cocina y haz clic '
        'en cualquier restaurante para descubrir los más similares, '
        'calculados mediante clustering KMeans sobre el perfil completo del local.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Buscador por nombre (filtra sobre _name_str, columna precalculada en loader.py)
    name_query = st.text_input(
        "buscar", placeholder="🔍  Busca un restaurante por nombre…",
        key="name_query", label_visibility="collapsed",
    )
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    filters = build_filters(encode, top_cuisines, country_cities, prefix="")
    st.markdown('<div class="rf-divider"></div>', unsafe_allow_html=True)

    filtered = apply_filters(df, encode, cuisine_idx, filters, name_query)
    total    = filtered.height

    # Estadísticas rápidas sobre el subconjunto filtrado
    avg_r   = filtered["avg_rating"].mean()
    avg_r   = avg_r * 0.5 + 1.0 if avg_r is not None else 0.0  # desnormalizar al rango 1-5
    pct_veg = (filtered["vegetarian_friendly"] == 1).sum() / max(total, 1) * 100

    st.markdown(
        f'<div class="rf-stats">'
        f'<div><span class="rf-stat-n">{total:,}</span>'
        f'<span class="rf-stat-l">restaurantes</span></div>'
        f'<div><span class="rf-stat-n">{avg_r:.1f}</span>'
        f'<span class="rf-stat-l">rating medio</span></div>'
        f'<div><span class="rf-stat-n">{pct_veg:.0f}%</span>'
        f'<span class="rf-stat-l">vegetarian-friendly</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if total == 0:
        st.info("No hay restaurantes con esos filtros.")
        return

    if total > DISPLAY_LIMIT:
        st.markdown(
            f'<div class="rf-caption">Mostrando los {DISPLAY_LIMIT} mejor valorados '
            f'de {total:,} resultados</div>',
            unsafe_allow_html=True,
        )

    display_df   = build_display_df(filtered, decode)
    # Las columnas internas (prefijo _) las ocultamos al usuario
    visible_cols = [c for c in display_df.columns if not c.startswith("_")]

    event = st.dataframe(
        display_df[visible_cols],
        selection_mode="single-row",
        on_select="rerun",
        use_container_width=True,
        hide_index=True,
        key="table",
    )

    selected = event.selection.rows
    if not selected:
        st.caption("Haz clic en un restaurante para ver recomendaciones similares.")
        return

    idx = selected[0]
    # Comprobamos que el índice sigue siendo válido después de cambiar los filtros
    if idx >= len(display_df):
        st.caption("Haz clic en un restaurante para ver recomendaciones similares.")
        return

    row_id  = display_df["_row_id"].iloc[idx]
    cluster = display_df["_cluster"].iloc[idx]
    name    = display_df["Nombre"].iloc[idx]

    st.markdown('<div class="rf-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="rf-rec-name">Similar a <em>{name}</em></div>'
        f'<div class="rf-rec-sub">Grupo {int(cluster)} · ordenado por similitud de perfil</div>',
        unsafe_allow_html=True,
    )

    recs = get_recommendations(df, row_id, cluster, filters, encode, cuisine_idx)
    if recs.height == 0:
        st.info("No hay recomendaciones con los filtros actuales.")
        return

    rec_df = build_rec_df(recs, decode)

    # Layout: ficha del restaurante seleccionado a la izquierda, recomendaciones a la derecha
    col_det, col_rec = st.columns([1, 2], gap="large")

    with col_det:
        row_data  = df.filter(pl.col("row_id") == row_id).row(0, named=True)
        city_name = decode["city"].get(int(row_data["city"]), "—")
        diets     = [
            label for label, col in (
                ("Vegetarian",  "vegetarian_friendly"),
                ("Vegan",       "vegan_options"),
                ("Gluten-free", "gluten_free"),
            ) if row_data[col] == 1
        ]
        st.markdown(
            f'<div class="rf-card"><table>'
            f'<tr><td class="lbl">País</td><td class="val">{display_df["País"].iloc[idx]}</td></tr>'
            f'<tr><td class="lbl">Ciudad</td><td class="val">{city_name}</td></tr>'
            f'<tr><td class="lbl">Rating</td><td class="val">{display_df["Rating"].iloc[idx]} / 5</td></tr>'
            f'<tr><td class="lbl">Precio</td><td class="val">{display_df["Precio"].iloc[idx]}</td></tr>'
            f'<tr><td class="lbl">Dieta</td><td class="val">{", ".join(diets) if diets else "—"}</td></tr>'
            f'<tr><td class="lbl">Grupo</td><td class="val">{int(cluster)}</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    with col_rec:
        rec_visible = [c for c in rec_df.columns if not c.startswith("_")]
        st.dataframe(
            rec_df[rec_visible],
            use_container_width=True,
            hide_index=True,
            column_config={
                # Similitud se muestra como barra de progreso (0-100%)
                "Similitud": st.column_config.ProgressColumn(
                    "Similitud", format="%d%%", min_value=0, max_value=100
                )
            },
        )
