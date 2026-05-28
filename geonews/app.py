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

#sidebar filters
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


# --- Cache data fetches so clicking doesn't re-query database ---
@st.cache_data
def load_louvain(location_type, scope_name, min_articles):
    return client.louvain(location_type, scope_name, min_articles)

@st.cache_data
def load_enrichment(location_type, scope_name):
    return client.enrichment(location_type, scope_name)

@st.cache_data
def load_temporal(location_type, scope_name):
    return client.temporal(location_type, scope_name)


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
tab1, tab2, tab3 = st.tabs([
    "Louvain",
    #"UMAP",
    #"K-means",
    #"Sankey",
    "Časovnica",
    #"Nomogram",
    "Enrichment"
])

with tab1:
    st.subheader("Geopolitične skupnosti lokacij")
    st.write("Lokacije so pobarvane glede na skupnost (Louvain algoritem).")

    min_articles = st.slider(
        "Minimalno število člankov za upoštevanje lokacije",
        min_value=1, max_value=100, value=10, step=1
    )

    with st.spinner("Nalagam Louvain..."):
        try:
            df_louvain = load_louvain(location_type, scope_name, min_articles)
        except Exception as e:
            st.error(f"Napaka: {e}")
            df_louvain = None

    if df_louvain is not None and not df_louvain.empty:

        # Build community → locations mapping
        community_locations = {}
        for _, row in df_louvain.iterrows():
            comm = int(row["community"])
            if comm not in community_locations:
                community_locations[comm] = set()
            community_locations[comm].add(row["source"])
            community_locations[comm].add(row["target"])

        sorted_communities = sorted(
            community_locations.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # Replicate the same colors as plot.py uses
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        unique_comms = sorted(community_locations.keys())
        cmap = plt.colormaps.get_cmap('tab20')
        community_colors = {
            c: mcolors.to_hex(cmap(unique_comms.index(c) / max(1, len(unique_comms) - 1)))
            for c in unique_comms
        }

        # Split into map (left) and community list (right)
        col_map, col_communities = st.columns([3, 1])

        with col_map:
            plot.louvain(df_louvain, "/tmp/louvain.html", color_mode=color_mode)
            with open("/tmp/louvain.html", "r") as f:
                components.html(f.read(), height=700, scrolling=True)

        with col_communities:
            st.subheader("Skupnosti")
            if "selected_community" not in st.session_state:
                st.session_state.selected_community = None

            for comm_id, locations in sorted_communities:
                color = community_colors[comm_id]
                
                # Count how many times each location appears as source or target
                location_counts = {}
                for _, row in df_louvain[df_louvain["community"] == comm_id].iterrows():
                    location_counts[row["source"]] = location_counts.get(row["source"], 0) + 1
                    location_counts[row["target"]] = location_counts.get(row["target"], 0) + 1
                # Sort by count, take top 3
                top3_list = sorted(location_counts, key=location_counts.get, reverse=True)[:3]
                top3 = ", ".join(top3_list)

                st.markdown(
                    f'<span style="color:{color}; font-size:22px;">■</span> '
                    f'**{top3}**<br>'
                    f'<small style="color:gray;">{len(locations)} lokacij</small>',
                    unsafe_allow_html=True
                )
                if st.button("Izberi", key=f"comm_{comm_id}"):
                    st.session_state.selected_community = comm_id
                st.markdown("---")

        # Details below when community is selected
        if st.session_state.selected_community is not None and st.session_state.selected_community in community_locations:
            sel = st.session_state.selected_community
            sel_locations = community_locations.get(sel, set())
            color = community_colors[sel]

            st.markdown(
                f'<h3><span style="color:{color};">■</span> Skupnost {sel} — podrobnosti</h3>',
                unsafe_allow_html=True
            )
            st.write(f"**Lokacije ({len(sel_locations)}):** {', '.join(sorted(sel_locations))}")

            st.subheader("Značilne teme")
            with st.spinner("Nalagam..."):
                try:
                    df_enrich = load_enrichment(location_type, scope_name)
                    if scope_name == "Republic of Slovenia":
                        df_enrich = df_enrich[df_enrich["topic"] != "slovenija"]
                    df_enrich_filtered = df_enrich[df_enrich["name"].isin(sel_locations)]
                    if df_enrich_filtered.empty:
                        st.info("Ni značilnih tem za to skupnost.")
                    else:
                        plot.enrichment(df_enrich_filtered, "/tmp/enrichment_comm.html", color_mode=color_mode)
                        with open("/tmp/enrichment_comm.html", "r") as f:
                            components.html(f.read(), height=500, scrolling=True)
                except Exception as e:
                    st.error(f"Napaka: {e}")

            st.subheader("Časovnica tem")
            with st.spinner("Nalagam..."):
                try:
                    df_temp = load_temporal(location_type, scope_name)
                    if scope_name == "Republic of Slovenia":
                        df_temp = df_temp[df_temp["topic"] != "slovenija"]
                    if df_temp.empty:
                        st.info("Ni podatkov.")
                    else:
                        plot.temporal(df_temp, "/tmp/temporal_comm.html", color_mode=color_mode)
                        with open("/tmp/temporal_comm.html", "r") as f:
                            components.html(f.read(), height=500, scrolling=True)
                except Exception as e:
                    st.error(f"Napaka: {e}")
    else:
        st.warning("Ni podatkov za izbrane nastavitve.")



with tab2:
    st.subheader("Razvoj tem skozi čas")
    with st.spinner("Nalagam..."):
        try:
            df = client.temporal(location_type, scope_name)
            if df.empty:
                st.warning("Ni podatkov za izbrane nastavitve.")
            else:
                if scope_name == "Republic of Slovenia":
                    df = df[df["topic"] != "slovenija"]
                plot.temporal(df, "/tmp/temporal.html", color_mode=color_mode)
                with open("/tmp/temporal.html", "r") as f:
                    components.html(f.read(), height=700, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")

with tab3:
    st.subheader("Statistično značilne teme po lokacijah")
    with st.spinner("Nalagam..."):
        try:
            df = client.enrichment(location_type, scope_name)
            if df.empty:
                st.warning("Ni podatkov za izbrane nastavitve.")
            else:
                if scope_name == "Republic of Slovenia":
                    df = df[df["topic"] != "slovenija"]
                plot.enrichment(df, "/tmp/enrichment.html", color_mode=color_mode)
                with open("/tmp/enrichment.html", "r") as f:
                    components.html(f.read(), height=700, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")