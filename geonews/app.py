import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from geonews.db import Client
from geonews import plot

#prebere .env file
load_dotenv()

st.set_page_config(layout="wide", page_title="Geografska analiza MMC")
st.title("Geografska analiza MMC člankov")

#poveze se na neo4j 
@st.cache_resource #poveze se samo enkrat, ne vsakic ko nekaj kliknem
def get_client():
    return Client(
        os.getenv("DB_ENDPOINT"),
        os.getenv("DB_USER"),
        os.getenv("DB_PASSWORD")
    )

client = get_client()

# --- Sidebar filters ---
st.sidebar.header("Nastavitve")

location_type = st.sidebar.selectbox(
    "Vrsta lokacije",
    ["Settlement", "SovereignState"],
    format_func=lambda x: "Naselja" if x == "Settlement" else "Države"
)

scope_options = ["Cel svet", "Republic of Slovenia", "Europe", "Asia", "United States"]
preset = st.sidebar.selectbox("Izberi območje", scope_options)
custom = st.sidebar.text_input("Ali vnesi po meri", placeholder="npr. Germany, Africa...")
scope_name = custom if custom else (None if preset == "Cel svet" else preset)

color_mode = st.sidebar.radio("Tema", ["dark", "light"])

st.sidebar.markdown("---")
st.sidebar.info(f"**Trenutne nastavitve:**\n\n Lokacije: {location_type}\n\n Območje: {scope_name or 'Cel svet'}")

# --- Helper to save and display html ---
def show_plot(plot_func, client_func, tmp_path, **kwargs):
    with st.spinner("Nalagam..."):
        try:
            df = client_func()
            if df.empty:
                st.warning("Ni podatkov za izbrane nastavitve.")
                return
            plot_func(df, tmp_path, color_mode=color_mode, **kwargs)
            with open(tmp_path, "r") as f:
                components.html(f.read(), height=700, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Louvain",
    "UMAP",
    "K-means",
    "Sankey",
    "Časovnica",
    "Nomogram",
    "Enrichment"
])

with tab1:
    st.subheader("Geopolitične skupnosti lokacij")
    st.write("Lokacije so pobarvane glede na skupnost (Louvain algoritem).")
    show_plot(
        plot.louvain,
        lambda: client.louvain(location_type, scope_name),
        "/tmp/louvain.html"
    )

with tab2:
    st.subheader("UMAP semantični prostor")
    st.write("Lokacije so razporejene glede na podobnost tem in sentimenta.")
    show_plot(
        plot.umap,
        lambda: client.umap(location_type, scope_name),
        "/tmp/umap.html"
    )

with tab3:
    st.subheader("K-means gruče")
    plot_type = st.radio("Prikaz", ["scatter", "map"], horizontal=True)
    with st.spinner("Nalagam..."):
        try:
            df = client.kmeans(location_type, scope_name)
            if df.empty:
                st.warning("Ni podatkov.")
            else:
                plot.kmeans(df, "/tmp/kmeans.html", plot_type, color_mode=color_mode)
                with open("/tmp/kmeans.html", "r") as f:
                    components.html(f.read(), height=600, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")

with tab4:
    st.subheader("Tok informacij: lokacija -> tema -> sentiment")
    top_n = st.slider("Število lokacij", 3, 15, 5)
    with st.spinner("Nalagam..."):
        try:
            df = client.sankey(location_type, scope_name)
            if df.empty:
                st.warning("Ni podatkov.")
            else:
                plot.sankey(df, "/tmp/sankey.html", top_n_locations=top_n, color_mode=color_mode)
                with open("/tmp/sankey.html", "r") as f:
                    components.html(f.read(), height=600, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")

with tab5:
    st.subheader("Razvoj tem skozi čas")
    show_plot(
        plot.temporal,
        lambda: client.temporal(location_type, scope_name),
        "/tmp/temporal.html"
    )

with tab6:
    st.subheader("Kaj vpliva na število komentarjev?")
    show_plot(
        plot.nomogram,
        lambda: client.lasso_regression(location_type, scope_name),
        "/tmp/nomogram.html"
    )

with tab7:
    st.subheader("Statistično značilne teme po lokacijah")
    show_plot(
        plot.enrichment,
        lambda: client.enrichment(location_type, scope_name),
        "/tmp/enrichment.html"
    )