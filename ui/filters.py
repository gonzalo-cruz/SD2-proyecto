# Widgets de filtrado y lógica de filtrado sobre el DataFrame de Polars.
# build_filters renderiza los controles en la UI; apply_filters aplica las
# condiciones al DataFrame usando máscaras booleanas de Polars.
import polars as pl
import streamlit as st


def build_filters(encode, top_cuisines, country_cities, prefix=""):
    # El prefijo permite usar los mismos filtros en dos tabs sin que los
    # widgets compartan clave y entren en conflicto (Streamlit obliga a
    # que cada widget tenga un key único por sesión)
    pk     = lambda k: f"{prefix}{k}"
    prices = ["Todos"] + sorted(encode["price_level"],
                                key=lambda p: encode["price_level"][p])
    countries = ["Todos"] + sorted(encode["country"])

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])

    country = c1.selectbox("País", countries, key=pk("country"))

    # Las ciudades disponibles dependen del país seleccionado
    if country != "Todos":
        cc           = encode["country"][country]
        city_options = country_cities.get(cc, [])
        city_ph      = "Todas las ciudades"
    else:
        city_options = []
        city_ph      = "Elige un país primero"
    city = c2.multiselect("Ciudad", city_options, key=pk("city"), placeholder=city_ph)

    min_rating = c3.slider("Rating mínimo", 1.0, 5.0, 1.0, step=0.5, key=pk("rating"))
    price      = c4.selectbox("Precio", prices, key=pk("price"))

    c5.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
    # Al resetear limpiamos también la selección de tabla para evitar índices desactualizados
    reset_keys = [pk(k) for k in ("country", "city", "rating", "price",
                                   "cuisines", "veg", "vegan", "gf")]
    if prefix == "":
        reset_keys += ["name_query", "table"]
    else:
        reset_keys += [pk("name_query"), "live_rid", "live_cluster", "live_name"]
    if c5.button("↺", key=pk("reset"), use_container_width=True, help="Restablecer filtros"):
        for k in reset_keys:
            st.session_state.pop(k, None)
        st.rerun()

    d1, d2, d3, d4 = st.columns([4, 1.2, 1.2, 1.2])
    cuisines = d1.multiselect("Cocina", top_cuisines, key=pk("cuisines"),
                               placeholder="Cualquier cocina")
    d2.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    veg   = d2.checkbox("Vegetariano", key=pk("veg"))
    d3.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    vegan = d3.checkbox("Vegano",      key=pk("vegan"))
    d4.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    gf    = d4.checkbox("Sin gluten",  key=pk("gf"))

    return dict(country=country, city=city, min_rating=min_rating, price=price,
                cuisines=cuisines, veg=veg, vegan=vegan, gf=gf)


def apply_filters(df, encode, cuisine_idx, f, name_query=""):
    # Construimos una máscara booleana acumulada; empezamos con True (sin filtro)
    mask = pl.lit(True)

    if f["country"] != "Todos":
        mask = mask & (pl.col("country") == encode["country"][f["country"]])

    if f.get("city"):
        # Convertimos los nombres de ciudad a sus códigos numéricos del encoding
        city_codes = {encode["city"][c] for c in f["city"] if c in encode["city"]}
        if city_codes:
            mask = mask & (pl.col("city").is_in(city_codes))

    if f["price"] != "Todos":
        mask = mask & (pl.col("price_level") == encode["price_level"][f["price"]])

    if f["min_rating"] > 1.0:
        min_code = encode["avg_rating"][str(f["min_rating"])]
        mask = mask & (pl.col("avg_rating") >= min_code)

    if f["cuisines"]:
        # cuisine_idx mapea cada tipo de cocina al conjunto de row_ids que la tienen
        valid: set = set()
        for c in f["cuisines"]:
            valid.update(cuisine_idx.get(c, set()))
        mask = mask & (pl.col("row_id").is_in(valid))

    if f["veg"]:
        mask = mask & (pl.col("vegetarian_friendly") == 1)
    if f["vegan"]:
        mask = mask & (pl.col("vegan_options") == 1)
    if f["gf"]:
        mask = mask & (pl.col("gluten_free") == 1)

    result = df.filter(mask)

    # Búsqueda por nombre: se aplica después de los filtros numéricos
    if name_query.strip():
        q      = name_query.strip().lower()
        result = result.filter(
            pl.col("_name_str").str.to_lowercase().str.contains(q, literal=True)
        )

    return result
