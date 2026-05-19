# Punto de entrada de la aplicación Streamlit.
# Se encarga únicamente de la configuración de página, cargar los datos
# y repartir el renderizado entre las dos tabs.
import streamlit as st

from loader import load_data
from pages.explore import render_explore
from pages.live import render_live
from ui.styles import CSS


def main():
    st.set_page_config(
        layout="wide",
        page_title="RestaurantFindr",
        initial_sidebar_state="collapsed",
    )
    # Inyectamos los estilos globales (fuente, colores, componentes)
    st.markdown(CSS, unsafe_allow_html=True)

    data = load_data()
    if data[0] is None:
        st.error("Ejecuta `python train_model.py` antes de lanzar la app.")
        return

    df, decode, encode, cuisine_idx, top_cuisines, country_cities = data

    tab_explore, tab_live = st.tabs(["Explorar", "En vivo"])

    with tab_explore:
        render_explore(df, decode, encode, cuisine_idx, top_cuisines, country_cities)

    with tab_live:
        render_live(df, decode, encode, cuisine_idx, top_cuisines, country_cities)


if __name__ == "__main__":
    main()
