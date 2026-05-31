import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from geonews.db import Client
from geonews import plot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Initialize session state
if "selected_community" not in st.session_state:
    st.session_state.selected_community = None
if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = None

#sidebar filters
st.sidebar.header("Nastavitve")

location_type = st.sidebar.selectbox(
    "Vrsta lokacije",
    ["Settlement", "SovereignState", "AdministrativeDivision", "Continent"],
    format_func=lambda x: {
        "Settlement": "Naselja",
        "SovereignState": "Države",
        "AdministrativeDivision": "Administrativne enote",
        "Continent": "Kontinenti"
    }[x]
)

scope_options = ["Cel svet", "Republic of Slovenia", "Europe", "Asia", "United States"]
preset = st.sidebar.selectbox("Izberi območje", scope_options)

st.sidebar.markdown("---")
st.sidebar.header("Iskanje lokacije")
search_query = st.sidebar.text_input("Išči po imenu", placeholder="npr. Slovenija, Berlin...")

if search_query:
    results = client.search_locations(search_query, top_k=5)
    if results.empty:
        st.sidebar.warning("Ni rezultatov.")
    else:
        st.sidebar.write("Najboljši zadetki:")
        for i, row in results.iterrows():
            label = f"{row['name']} ({row['type']})"
            # Use index i in key to avoid duplicates
            if st.sidebar.button(label, key=f"search_{i}_{row['name']}"):
                st.session_state.search_selected = row['name']
                st.session_state.search_type = row['type']
                st.rerun()

# Use search result if selected, otherwise use preset
if "search_selected" in st.session_state and st.session_state.search_selected:
    scope_name = st.session_state.search_selected
else:
    scope_name = None if preset == "Cel svet" else preset

color_mode = st.sidebar.radio("Tema", ["dark", "light"])

st.sidebar.markdown("---")
st.sidebar.info(f"**Trenutne nastavitve:**\n\n Lokacije: {location_type}\n\n Območje: {scope_name or 'Cel svet'}")


# --- Cache data fetches so clicking doesn't re-query database ---
@st.cache_data
def load_louvain(location_type, scope_name, min_articles, tolerance):
    return client.louvain(location_type, scope_name, min_articles, tolerance)

@st.cache_data
def load_enrichment(location_type, scope_name):
    return client.enrichment(location_type, scope_name)

@st.cache_data
def load_temporal(location_type, scope_name):
    return client.temporal(location_type, scope_name)

@st.cache_data
def load_articles_for_location(location_name):
    return client.articles_for_location(location_name)

@st.cache_data
def load_cluster_keywords(location_names_tuple):
    return client.cluster_keywords(list(location_names_tuple))

@st.cache_data
def load_cluster_sentiment(location_names_tuple):
    return client.cluster_sentiment(list(location_names_tuple))

@st.cache_data
def load_kmeans_keywords(location_type, scope_name, k_clusters, min_articles):
    return client.kmeans_keywords(location_type, scope_name, k_clusters, min_articles)


def get_map_focus(scope_name):
    """Returns center lat, lon and zoom scale for a given scope."""
    if not scope_name:
        return (20, 10, 1)  # whole world
    
    # Hardcoded zoom levels for known scopes
    zoom_map = {
        "Europe": 2.5,
        "Asia": 1.5,
        "Africa": 1.5,
        "North America": 1.5,
        "South America": 1.5,
    }
    
    # Get coordinates from database
    coords = client.get_location_center(scope_name)
    if coords is None:
        return (20, 10, 1)
    
    lat, lon = coords
    # Use hardcoded zoom if available, otherwise guess based on location type
    zoom = zoom_map.get(scope_name, 6)  # default zoom 6 for countries
    
    return (lat, lon, zoom)


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
tab1, tab2, tab3, tab4 = st.tabs([
    "Louvain",
    "KMeans",
    "Časovnica",
    "Enrichment"
])

with tab1:
    st.subheader("Geopolitične gruče lokacij")
    st.write("Lokacije so pobarvane glede na gručo (Louvain algoritem).")

    min_articles = st.slider(
        "Minimalno število člankov za upoštevanje lokacije",
        min_value=1, max_value=100, value=10, step=1
    )

    tolerance = st.select_slider(
        "Število skupnosti (višje = več skupnosti)",
        options=[0.00001, 0.0001, 0.001, 0.01, 0.1],
        value=0.0001
    )

    with st.spinner("Nalagam Louvain..."):
        try:
            df_louvain = load_louvain(location_type, scope_name, min_articles, tolerance)
        except Exception as e:
            st.error(f"Napaka: {e}")
            df_louvain = None

    if df_louvain is not None and not df_louvain.empty:
        if "selected_community" not in st.session_state:
            st.session_state.selected_community = None

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

        # Clear button FIRST
        if st.session_state.get("selected_community") is not None:
            col_back, col_info = st.columns([1, 4])
            with col_back:
                if st.button("<- Nazaj"):
                    st.session_state.selected_community = None
                    st.session_state.last_clicked = None
                    st.rerun()
            with col_info:
                st.info(f"Izbrana skupnost — klikni 'Nazaj' za prikaz vseh")

        # Show map
        chart_key = f"louvain_map_{st.session_state.get('selected_community')}_{st.session_state.get('last_clicked')}"
        center_lat, center_lon, proj_scale = get_map_focus(scope_name)
        fig = plot.louvain_figure(
            df_louvain,
            color_mode=color_mode,
            selected_community=st.session_state.get("selected_community"),
            center_lat=center_lat,
            center_lon=center_lon,
            projection_scale=proj_scale
        )
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True, key=chart_key)

        # Handle click
        if event and event.selection and event.selection.points:
            clicked_location = event.selection.points[0].get("text")
            if clicked_location and clicked_location != st.session_state.get("last_clicked"):
                st.session_state.last_clicked = clicked_location
                for comm_id, locations in community_locations.items():
                    if clicked_location in locations:
                        st.session_state.selected_community = comm_id
                        st.rerun()
                        break
        

        # Details below when community is selected
        if st.session_state.selected_community is not None and st.session_state.selected_community in community_locations:
            sel = st.session_state.selected_community
            sel_locations = community_locations.get(sel, set())
            sel_locations_tuple = tuple(sorted(sel_locations))
            color = community_colors[sel]

            st.markdown(
                f'<h3><span style="color:{color};">■</span> Skupnost — podrobnosti</h3>',
                unsafe_allow_html=True
            )
            st.write(f"**Lokacije ({len(sel_locations)}):**")

            # Keywords and Sentiment side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ključne besede")
                try:
                    import json
                    tfidf_data = json.loads(cluster_data['tfidf_keywords'].iloc[0])
                    df_kw = pd.DataFrame(tfidf_data)
                    df_kw = df_kw[df_kw['score'] > 0].sort_values('score')
                    fig_kw = px.bar(
                        df_kw,
                        x="score", y="keyword",
                        orientation="h",
                        labels={"score": "TF-IDF distinctive score", "keyword": ""},
                        template="plotly_dark" if not is_light else "plotly_white",
                        color="score",
                        color_continuous_scale="Viridis"
                    )
                    fig_kw.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                    st.plotly_chart(fig_kw, use_container_width=True)
                except Exception as e:
                    st.error(f"Napaka: {e}")

            with col2:
                st.subheader("Sentiment skupnosti")
                st.write("Primerjava sentimenta te skupnosti z globalnim povprečjem.")
                with st.spinner("Nalagam sentiment..."):
                    try:
                        df_sent = load_cluster_sentiment(sel_locations_tuple)
                        if df_sent.empty:
                            st.info("Ni podatkov o sentimentu.")
                        else:
                            import plotly.graph_objects as go
                            fig_sent = go.Figure()
                            fig_sent.add_trace(go.Bar(
                                name="Ta skupnost",
                                x=df_sent["sentiment"],
                                y=df_sent["cluster_pct"],
                                marker_color="#4e79a7"
                            ))
                            fig_sent.add_trace(go.Bar(
                                name="Globalno povprečje",
                                x=df_sent["sentiment"],
                                y=df_sent["global_pct"],
                                marker_color="#f28e2b"
                            ))
                            fig_sent.update_layout(
                                barmode="group",
                                title="Sentiment: skupnost vs. globalno",
                                yaxis_title="Delež člankov (%)",
                                template="plotly_dark" if color_mode == "dark" else "plotly_white",
                                height=400,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(fig_sent, use_container_width=True)
                    except Exception as e:
                        st.error(f"Napaka: {e}")

            st.markdown("---")

            #Location selector and articles
            selected_location = st.selectbox(
                "Izberi lokacijo za prikaz člankov",
                options=["— izberi lokacijo —"] + sorted(sel_locations)
            )

            if selected_location != "— izberi lokacijo —":
                st.markdown("---")
                st.subheader(f"Članki o: {selected_location}")

                with st.spinner("Nalagam članke..."):
                    df_articles = load_articles_for_location(selected_location)

                if df_articles.empty:
                    st.info("Ni člankov za to lokacijo.")
                else:
                    col_f1, col_f2, col_f3 = st.columns(3)

                    with col_f1:
                        topics = ["Vse teme"] + sorted(df_articles["topic"].dropna().unique().tolist())
                        selected_topic = st.selectbox("Tema", topics)

                    with col_f2:
                        sentiments = ["Vsi sentimenti"] + sorted(df_articles["sentiment"].dropna().unique().tolist())
                        selected_sentiment = st.selectbox("Sentiment", sentiments)

                    with col_f3:
                        sort_by = st.selectbox("Razvrsti po", ["Datum (najnovejši)", "Datum (najstarejši)", "Največ komentarjev"])

                    df_filtered = df_articles.copy()
                    if selected_topic != "Vse teme":
                        df_filtered = df_filtered[df_filtered["topic"] == selected_topic]
                    if selected_sentiment != "Vsi sentimenti":
                        df_filtered = df_filtered[df_filtered["sentiment"] == selected_sentiment]

                    if sort_by == "Datum (najnovejši)":
                        df_filtered = df_filtered.sort_values("date", ascending=False)
                    elif sort_by == "Datum (najstarejši)":
                        df_filtered = df_filtered.sort_values("date", ascending=True)
                    elif sort_by == "Največ komentarjev":
                        df_filtered = df_filtered.sort_values("n_comments", ascending=False)

                    df_filtered = df_filtered.drop_duplicates(subset=["id"])

                    total = len(df_filtered)
                    show_all = st.checkbox("Prikaži vse članke")
                    if not show_all:
                        top_n_articles = st.slider("Število člankov", min_value=5, max_value=min(100, total), value=10, step=5)
                        df_filtered = df_filtered.head(top_n_articles)

                    st.write(f"**Prikazujem {len(df_filtered)} od {total} člankov**")
                    st.markdown("---")

                    for _, article in df_filtered.iterrows():
                        col_text = st.container()
                        with col_text:
                            st.markdown(f"### [{article['title']}]({article['url']})")
                            date_str = article['date'].strftime("%d. %m. %Y") if pd.notna(article['date']) else "neznan datum"
                            comments = int(article['n_comments']) if pd.notna(article['n_comments']) else 0
                            topic_str = article['topic'] if pd.notna(article['topic']) else "/"
                            sentiment_str = article['sentiment'] if pd.notna(article['sentiment']) else "/"
                            st.markdown(
                                f"{date_str} &nbsp;|&nbsp; "
                                f"{comments} komentarjev &nbsp;|&nbsp; "
                                f"{topic_str} &nbsp;|&nbsp; "
                                f"{sentiment_str}"
                            )
                        st.markdown("---")

            st.markdown("---")

            # Enrichment and Timeline
            st.subheader("Značilne teme")
            with st.spinner("Nalagam..."):
                try:
                    df_enrich = load_enrichment(location_type, scope_name)
                    df_enrich = df_enrich[~((df_enrich["name"] == "Republic of Slovenia") & (df_enrich["topic"] == "slovenija"))]
                    if scope_name and "slovenia" in scope_name.lower():
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
                    if scope_name and "slovenia" in scope_name.lower():
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
    st.subheader("Semantične gruče lokacij")
    st.write("Lokacije so združene glede na podobnost ključnih besed v novicah — ne glede na geografijo.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        min_articles_km = st.slider(
            "Minimalno število člankov",
            min_value=1, max_value=50, value=5, step=1,
            key="km_min_articles"
        )
    with col_s2:
        n_clusters_km = st.slider(
            "Število gruč",
            min_value=2, max_value=10, value=4, step=1,
            key="km_n_clusters"
        )

    with st.spinner("Računam semantične gruče..."):
        try:
            df_km = load_kmeans_keywords(
                location_type, scope_name, n_clusters_km, min_articles_km
            )

            if df_km is None or df_km.empty:
                st.warning("Ni dovolj podatkov.")
            else:
                # Initialize session state for kmeans
                if "selected_km_cluster" not in st.session_state:
                    st.session_state.selected_km_cluster = None
                if "last_clicked_km" not in st.session_state:
                    st.session_state.last_clicked_km = None

                # Back button
                if st.session_state.selected_km_cluster is not None:
                    if st.button("← Nazaj na vse gruče", key="km_back"):
                        st.session_state.selected_km_cluster = None
                        st.session_state.last_clicked_km = None
                        st.rerun()

                is_light = color_mode == "light"
                # Build figure with opacity based on selected cluster
                selected_km = st.session_state.selected_km_cluster

                fig_km = go.Figure()

                colors = px.colors.qualitative.Bold
                cluster_ids = sorted(df_km['cluster_id'].unique())
                color_map = {c: colors[i % len(colors)] for i, c in enumerate(cluster_ids)}

                center_lat, center_lon, proj_scale = get_map_focus(scope_name)
                fig_km = plot.kmeans_figure(
                    df_km,
                    color_mode=color_mode,
                    selected_cluster=st.session_state.selected_km_cluster,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    projection_scale=proj_scale
                )

                km_event = st.plotly_chart(
                    fig_km,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"km_chart_{st.session_state.selected_km_cluster}_{st.session_state.last_clicked_km}"
                )

                # Handle click
                if km_event and km_event.selection and km_event.selection.points:
                    clicked = km_event.selection.points[0]
                    clicked_name = clicked.get("text") or clicked.get("hovertext")
                    if clicked_name and clicked_name != st.session_state.last_clicked_km:
                        st.session_state.last_clicked_km = clicked_name
                        # Find cluster for this location
                        match = df_km[df_km['name'] == clicked_name]
                        if not match.empty:
                            st.session_state.selected_km_cluster = match.iloc[0]['cluster_id']
                            st.rerun()

                # Show cluster details when selected
                if st.session_state.selected_km_cluster is not None:
                    cluster_data = df_km[df_km['cluster_id'] == st.session_state.selected_km_cluster]
                    keywords = cluster_data['cluster_keywords'].iloc[0]
                    locations = sorted(cluster_data['name'].tolist())

                    st.markdown("---")
                    st.subheader(f"Gruča: {keywords}")
                    st.write(f"**{len(locations)} lokacij:** {', '.join(locations)}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Ključne besede")
                        try:
                            import json
                            tfidf_data = json.loads(cluster_data['tfidf_keywords'].iloc[0])
                            df_kw = pd.DataFrame(tfidf_data)
                            df_kw = df_kw[df_kw['score'] > 0].sort_values('score')
                            fig_kw = px.bar(
                                df_kw,
                                x="score", y="keyword",
                                orientation="h",
                                labels={"score": "Indeks značilnosti (%)", "keyword": ""},
                                template="plotly_dark" if not is_light else "plotly_white",
                                color="score",
                                color_continuous_scale="Viridis"
                            )
                            fig_kw.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                            st.plotly_chart(fig_kw, use_container_width=True)
                        except Exception as e:
                            st.error(f"Napaka: {e}")

                    with col2:
                        st.subheader("Sentiment")
                        with st.spinner("Nalagam..."):
                            try:
                                df_sent = load_cluster_sentiment(tuple(sorted(locations)))
                                if not df_sent.empty:
                                    fig_sent = go.Figure()
                                    fig_sent.add_trace(go.Bar(
                                        name="Ta gruča",
                                        x=df_sent["sentiment"],
                                        y=df_sent["cluster_pct"],
                                        marker_color="#4e79a7"
                                    ))
                                    fig_sent.add_trace(go.Bar(
                                        name="Globalno",
                                        x=df_sent["sentiment"],
                                        y=df_sent["global_pct"],
                                        marker_color="#f28e2b"
                                    ))
                                    fig_sent.update_layout(
                                        barmode="group",
                                        yaxis_title="Delež člankov (%)",
                                        template="plotly_dark" if not is_light else "plotly_white",
                                        height=400,
                                        margin=dict(l=0, r=0, t=0, b=0)
                                    )
                                    st.plotly_chart(fig_sent, use_container_width=True)
                            except Exception as e:
                                st.error(f"Napaka: {e}")

                    # Article selector
                    st.markdown("---")
                    selected_location_km = st.selectbox(
                        "Izberi lokacijo za prikaz člankov",
                        options=["— izberi lokacijo —"] + locations,
                        key="km_location_select"
                    )

                    if selected_location_km != "— izberi lokacijo —":
                        st.subheader(f"Članki o: {selected_location_km}")
                        with st.spinner("Nalagam članke..."):
                            df_articles = load_articles_for_location(selected_location_km)
                        if df_articles.empty:
                            st.info("Ni člankov.")
                        else:
                            df_articles = df_articles.drop_duplicates(subset=["id"])
                            df_articles = df_articles.sort_values("date", ascending=False)
                            total = len(df_articles)
                            top_n = st.slider("Število člankov", min_value=5, max_value=min(100, total), value=10, step=5, key="km_articles_slider")
                            df_articles = df_articles.head(top_n)
                            st.write(f"**Prikazujem {len(df_articles)} od {total} člankov**")
                            st.markdown("---")
                            for _, article in df_articles.iterrows():
                                st.markdown(f"### [{article['title']}]({article['url']})")
                                date_str = article['date'].strftime("%d. %m. %Y") if pd.notna(article['date']) else "neznan datum"
                                comments = int(article['n_comments']) if pd.notna(article['n_comments']) else 0
                                topic_str = article['topic'] if pd.notna(article['topic']) else "/"
                                sentiment_str = article['sentiment'] if pd.notna(article['sentiment']) else "/"
                                st.markdown(f"{date_str} &nbsp;|&nbsp; {comments} komentarjev &nbsp;|&nbsp; {topic_str} &nbsp;|&nbsp; {sentiment_str}")
                                st.markdown("---")

        except Exception as e:
            st.error(f"Napaka: {e}")


with tab3:
    st.subheader("Razvoj tem skozi čas")
    with st.spinner("Nalagam..."):
        try:
            df = client.temporal(location_type, scope_name)
            if df.empty:
                st.warning("Ni podatkov za izbrane nastavitve.")
            else:
                if scope_name and "slovenia" in scope_name.lower():
                    df = df[df["topic"] != "slovenija"]
                plot.temporal(df, "/tmp/temporal.html", color_mode=color_mode)
                with open("/tmp/temporal.html", "r") as f:
                    components.html(f.read(), height=700, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")

with tab4:
    #st.write(f"DEBUG scope_name: {scope_name}")
    st.subheader("Statistično značilne teme po lokacijah")
    with st.spinner("Nalagam..."):
        try:
            df = client.enrichment(location_type, scope_name)
            df = df[~((df["name"] == "Republic of Slovenia") & (df["topic"] == "slovenija"))]
            if df.empty:
                st.warning("Ni podatkov za izbrane nastavitve.")
            else:
                if scope_name and "slovenia" in scope_name.lower():
                    df = df[df["topic"] != "slovenija"]
                plot.enrichment(df, "/tmp/enrichment.html", color_mode=color_mode)
                with open("/tmp/enrichment.html", "r") as f:
                    components.html(f.read(), height=700, scrolling=True)
        except Exception as e:
            st.error(f"Napaka: {e}")