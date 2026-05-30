"""Functions for creating visualizations."""

from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def louvain(
        df_network: pd.DataFrame, 
        output_filename: str = "geo_nodes.html", 
        max_nodes: int = 10000,
        min_node_size: int = 10,
        max_node_size: int = 50,
        node_scale_factor: float = 0.7,
        color_mode: Literal["dark", "light"] = "dark" 
        ):
    """
    Plot Louvain communities and nodes on a interactive world map.

    Args:
        df_network (pd.DataFrame): Table containing network data, generated using run_lovain().
        output_filename (str): Name of the .html file where to save the visual.
        max_nodes (int): Number of top nodes to plot.
        min_node_size (int): Minimum node size on the map.
        max_node_size (int): Maximum node size on the map.
        node_scale_factor (float): Factor for scaling node size based on importance.
        color_mode (Literal['dark', 'light']): Color mode of the map.
    """

    # Drop locations with no coordinates.
    for col in ['source_lat', 'source_lon', 'target_lat', 'target_lon']:
        df_network[col] = pd.to_numeric(df_network[col], errors='coerce')
    df_network = df_network.dropna(subset=['source_lat', 'source_lon', 'target_lat', 'target_lon'])
    
    # Build initial graph to calculate degree of nodes, and grab top nodes.
    full_G = nx.from_pandas_edgelist(df_network, 'source', 'target', ['weight', 'community'])
    node_importance = dict(full_G.degree())
    top_nodes = sorted(node_importance, key=node_importance.get, reverse=True)[:max_nodes] # type: ignore

    # Extract node data.
    node_data = {}
    for _, row in df_network.iterrows():
        if row['source'] in top_nodes and row['source'] not in node_data:
            node_data[row['source']] = {'lat': row['source_lat'], 'lon': row['source_lon'], 'community': row['community'], 'degree': node_importance[row['source']]}
        if row['target'] in top_nodes and row['target'] not in node_data:
            node_data[row['target']] = {'lat': row['target_lat'], 'lon': row['target_lon'], 'community': row['community'], 'degree': node_importance[row['target']]}

    # Prepare lists for Plotly.
    node_names = list(node_data.keys())
    node_lats = [data['lat'] for data in node_data.values()]
    node_lons = [data['lon'] for data in node_data.values()]
    node_communities = [data['community'] for data in node_data.values()] 
    
    # Create hover .html text.
    hover_texts = [f"<b>{name}</b><br>Connections (Degree): {data['degree']}<br>Community ID: {data['community']}" for name, data in node_data.items()]
    node_sizes = [max(min_node_size, min(data['degree'] * node_scale_factor, max_node_size)) for data in node_data.values()] 

    # Create color mapping.
    unique_comms = sorted(set(node_communities))
    cmap = plt.colormaps.get_cmap('tab20')
    plotly_colors = [mcolors.to_hex(cmap(unique_comms.index(c) / max(1, len(unique_comms) - 1))) for c in node_communities]

    # Create color mode.
    is_light = color_mode.lower() == "light"
    theme = {
        "text": "black" if is_light else "white",
        "bg": "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)",
        "land": "rgb(240, 240, 240)" if is_light else "rgb(40, 40, 40)",
        "ocean": "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)",
        "borders": "rgb(180, 180, 180)" if is_light else "rgb(80, 80, 80)"
    }

    # Plot nodes.
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=node_lons, lat=node_lats, mode='markers', text=hover_texts, hoverinfo='text',
        marker=dict(size=node_sizes, color=plotly_colors, line=dict(width=0.3, color=theme["bg"]), opacity=0.90)
    ))  

    # Configure plot.
    fig.update_layout(
        title=dict(text=f"Geopolitical News Hubs", font=dict(size=20, color=theme["text"])),
        showlegend=False,
        geo=dict(
            projection_type="natural earth", showland=True, showocean=True, showlakes=True, showcountries=True,
            landcolor=theme["land"], oceancolor=theme["ocean"], lakecolor=theme["ocean"], 
            countrycolor=theme["borders"], coastlinecolor=theme["borders"]
        ),
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"], margin=dict(l=0, r=0, t=50, b=0) 
    )

    fig.write_html(output_filename)

def louvain_figure(
        df_network: pd.DataFrame, 
        max_nodes: int = 10000,
        min_node_size: int = 10,
        max_node_size: int = 50,
        node_scale_factor: float = 0.7,
        color_mode: Literal["dark", "light"] = "dark",
        selected_community=None,
        center_lat: float = 20,
        center_lon: float = 10,
        projection_scale: float = 1
        ):
    """
    Same as louvain() but returns the Plotly figure instead of saving to HTML.
    Used for interactive click events in Streamlit.
    """
    for col in ['source_lat', 'source_lon', 'target_lat', 'target_lon']:
        df_network[col] = pd.to_numeric(df_network[col], errors='coerce')
    df_network = df_network.dropna(subset=['source_lat', 'source_lon', 'target_lat', 'target_lon'])
    
    full_G = nx.from_pandas_edgelist(df_network, 'source', 'target', ['weight', 'community'])
    node_importance = dict(full_G.degree())
    top_nodes = sorted(node_importance, key=node_importance.get, reverse=True)[:max_nodes]

    node_data = {}
    for _, row in df_network.iterrows():
        if row['source'] in top_nodes and row['source'] not in node_data:
            node_data[row['source']] = {'lat': row['source_lat'], 'lon': row['source_lon'], 'community': row['community'], 'degree': node_importance[row['source']]}
        if row['target'] in top_nodes and row['target'] not in node_data:
            node_data[row['target']] = {'lat': row['target_lat'], 'lon': row['target_lon'], 'community': row['community'], 'degree': node_importance[row['target']]}

    node_names = list(node_data.keys())
    node_lats = [data['lat'] for data in node_data.values()]
    node_lons = [data['lon'] for data in node_data.values()]
    node_communities = [data['community'] for data in node_data.values()]
    
    hover_texts = [f"<b>{name}</b><br>Connections: {data['degree']}<br>Community: {data['community']}" for name, data in node_data.items()]
    node_sizes = [max(min_node_size, min(data['degree'] * node_scale_factor, max_node_size)) for data in node_data.values()]
    if selected_community is not None:
        node_opacities = [0.9 if data['community'] == selected_community else 0.1 for data in node_data.values()]
    else:
        node_opacities = [0.9] * len(node_data)

    unique_comms = sorted(set(node_communities))
    cmap = plt.colormaps.get_cmap('tab20')
    plotly_colors = [mcolors.to_hex(cmap(unique_comms.index(c) / max(1, len(unique_comms) - 1))) for c in node_communities]

    is_light = color_mode.lower() == "light"
    theme = {
        "text": "black" if is_light else "white",
        "bg": "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)",
        "land": "rgb(240, 240, 240)" if is_light else "rgb(40, 40, 40)",
        "ocean": "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)",
        "borders": "rgb(180, 180, 180)" if is_light else "rgb(80, 80, 80)"
    }

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=node_lons, lat=node_lats, mode='markers',
        text=node_names,
        hovertext=hover_texts,
        hoverinfo='text',
        marker=dict(size=node_sizes, color=plotly_colors, line=dict(width=0.3, color=theme["bg"]), opacity=node_opacities),
        customdata=node_communities
    ))

    fig.update_layout(
        showlegend=False,
        height=600,
        geo=dict(
            projection_type="natural earth",
            showland=True, showocean=True, showlakes=True, showcountries=True,
            landcolor=theme["land"], oceancolor=theme["ocean"], lakecolor=theme["ocean"],
            countrycolor=theme["borders"], coastlinecolor=theme["borders"],
            lataxis_range=[-60, 90],
            lonaxis_range=[-180, 180],
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=projection_scale
        ),
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"], margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def sankey(
        df: pd.DataFrame, 
        output_filename: str = "sankey_flow.html", 
        top_n_locations: int = 5, 
        color_mode: Literal["dark", "light"] = "dark" 
    ):
    """
    Plot Sankey flow diagram: Location -> Topic -> Sentiment.

    Args:
        df (pd.DataFrame): The data.
        output_filename (str): Path to .html file where the plot will be saved.
        top_n_locations (int): Number of top locations to plot.
        color_mode (Literal['dark', 'light']): Color mode of the plot.
    """
    # Filter top locations.
    top_locs = df['name'].value_counts().nlargest(top_n_locations).index
    df_sankey = df[df['name'].isin(top_locs)].dropna(subset=['topic', 'sentiment'])
    if df_sankey.empty: return

    # Calculate flow weights for Location -> Topic.
    loc_topic = df_sankey.groupby(['name', 'topic']).size().reset_index(name='value')
    loc_topic.columns = ['source', 'target', 'value']
    
    # Calculate flow weights for Topic -> Sentiment.
    topic_sent = df_sankey.groupby(['topic', 'sentiment']).size().reset_index(name='value')
    topic_sent.columns = ['source', 'target', 'value']
    
    # Combine flows.
    links_df = pd.concat([loc_topic, topic_sent], ignore_index=True)
    
    # Create a unique index mapping for Plotly.
    all_nodes = list(pd.unique(links_df[['source', 'target']].values.ravel('K')))
    node_mapping = {node: i for i, node in enumerate(all_nodes)}
    
    links_df['source_idx'] = links_df['source'].map(node_mapping)
    links_df['target_idx'] = links_df['target'].map(node_mapping)
    
    # Configure theme.
    is_light = color_mode.lower() == "light"
    bg_col = "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)"
    text_col = "black" if is_light else "white"
    
    # Build the figure.
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 20,
            thickness = 25,
            line = dict(color = "white" if not is_light else "black", width = 0.5),
            label = all_nodes,
            color = px.colors.qualitative.Pastel * 3
        ),
        link = dict(
            source = links_df['source_idx'],
            target = links_df['target_idx'],
            value = links_df['value'],
            color = "rgba(180, 180, 180, 0.3)"
        )
    )])

    fig.update_layout(
        title=dict(text=f"Information Flow", font=dict(size=20, color=text_col)),
        font=dict(size=12, color=text_col),
        paper_bgcolor=bg_col,
        plot_bgcolor=bg_col
    )
    fig.write_html(output_filename)

def temporal(
        df_temp: pd.DataFrame, 
        output_filename: str = "temporal.html", 
        color_mode: Literal["dark", "light"] = "dark"
    ):
    """
    Plot time series showcasing evolution of topic-coverage through time.

    Args:
        df_temp (pd.DataFrame): The data.
        output_filename (str): Path to .html file where the plot will be saved.
        color_mode (Literal['dark', 'light']): Color theme of the plot.
    """    
    # Group by year-month, to smooth out daily spikes, and topic.
    df_temp['month_year'] = df_temp['publish_date'].dt.to_period('M').astype(str)
    time_series = df_temp.groupby(['month_year', 'topic']).size().reset_index(name='article_count')
    time_series = time_series.sort_values('month_year')
    
    # Configure theme.
    template_name = "plotly_white" if color_mode.lower() == "light" else "plotly_dark"
    
    # Build the figure.
    fig = px.area(
        time_series, 
        x="month_year", 
        y="article_count", 
        color="topic",
        title="Temporal Evolution of News Topics",
        labels={'month_year': 'Timeline', 'article_count': 'Number of Articles'},
        template=template_name,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Smooth the lines visually and remove the gaps between data points.
    fig.update_traces(line=dict(width=0), mode='lines')
    fig.update_layout(margin=dict(r=150))
    fig.write_html(output_filename)

def kmeans(
        df_clusters: pd.DataFrame, 
        output_filename: str = "clusters.html", 
        plot_type: Literal["map", "scatter"] = "scatter",
        color_mode: Literal["dark", "light"] = "dark"
    ):
    """
    Plot k-means clusters either as a scatter plot or on a world map.

    Args:
        df_clusters (pd.DataFrame): The data.
        output_filename (str): Path to .html file where plot will be saved.
        plot_type (Literal['map', 'scatter']): Whether to plot on world map or scatter plot.
        color_mode (Literal['dark', 'light']): Color mode of the plot.
    """
    
    # Configure color theme.
    is_light = color_mode.lower() == "light"
    template_name = "plotly_white" if is_light else "plotly_dark"
    land_col = "rgb(240, 240, 240)" if is_light else "rgb(40, 40, 40)"
    ocean_col = "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)"

    # Scatter plot.
    if plot_type == "scatter":
        fig = px.scatter(
            df_clusters, x='avg_sentiment', y='avg_comments', color='cluster_id', hover_name='name', size='article_count',
            title="K-Means Clusters: Sentiment vs. Engagement",
            labels={'avg_sentiment': 'Average Sentiment (-1 to 1)', 'avg_comments': 'Average Comments'},
            template=template_name, color_discrete_sequence=px.colors.qualitative.Pastel
        )
    
    # World map.
    else:
        fig = px.scatter_geo(
            df_clusters, lat='lat', lon='lon', color='cluster_id', hover_name='name', size='article_count',
            title="Global Zones by News Engagement & Sentiment",
            template=template_name, projection="natural earth",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_geos(showland=True, landcolor=land_col, showocean=True, oceancolor=ocean_col, showlakes=True, lakecolor=ocean_col)

    fig.write_html(output_filename)

def umap(
        df_semantic: pd.DataFrame, 
        output_filename: str = "umap.html", 
        color_mode: Literal["dark", "light"] = "dark"
    ):
    """
    Plot UMAP 2D visualization.

    Args:
        df_semantic (pd.DataFrame): The data.
        output_filename (str): Path to .html file where plot will be saved.
        color_mode (Literal['dark', 'light']): Color theme of the plot.
    """    
    # Configure color theme.
    is_light = color_mode.lower() == "light"
    template_name = "plotly_white" if is_light else "plotly_dark"
    land_col = "rgb(240, 240, 240)" if is_light else "rgb(40, 40, 40)"
    ocean_col = "rgb(255, 255, 255)" if is_light else "rgb(15, 15, 20)"

    # Create scatter plot.
    fig_scatter = px.scatter(
        df_semantic, x='umap_x', y='umap_y', hover_name='name', size='article_count',
        color='article_count', color_continuous_scale='Viridis',
        title="UMAP Semantic Space",
        template=template_name
    )
    fig_scatter.write_html(output_filename)

def enrichment(
        df_enrich: pd.DataFrame, 
        output_filename: str = "enrichment.html", 
        color_mode: Literal["dark", "light"] = "dark"
    ):
    """
    Plot barchart of enriched location-topic pairs.

    Args:
        df_enrich (pd.DataFrame): The data.
        output_filename (str): Path to .html file where plot will be saved.
        color_mode (Literal['dark', 'light']): Color mode of the plot.
    """
    # Configure color theme.
    is_light = color_mode.lower() == "light"
    template_name = "plotly_white" if is_light else "plotly_dark"
    
    # Fetch top 20 most enriched location-topic pairs.
    top_enrichments = df_enrich.head(20).copy()
    top_enrichments['Label'] = top_enrichments['name'] + " - " + top_enrichments['topic']
    
    # Plot barchart.
    fig_bar = px.bar(
        top_enrichments.sort_values('significance', ascending=True), 
        x='significance', y='Label', color='topic',
        title="Top Statistically Enriched Topics per Location (-log10 p-value)",
        labels={'significance': 'Statistical Significance (Higher = Less likely by chance)'},
        template=template_name, color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_bar.update_layout(height=700)
    fig_bar.write_html(output_filename)

def nomogram(
        coef_df: pd.DataFrame, 
        output_filename:str = "lasso_weights.html", 
        color_mode: Literal["dark", "light"] = "dark"
    ):
    """
    Plot nomogram for a trained regression model.

    Args:
        coef_df (pd.DataFrame): The data.
        output_filename (str): Path to .html file where plot will be saved.
        color_mode (Literal['dark', 'light']): Color mode of the plot.
    """
    # Configure color theme.
    is_light = color_mode.lower() == "light"
    template_name = "plotly_white" if is_light else "plotly_dark"
    
    # Fetch the top 15 positive predictors and top 15 negative predictors.
    top_pos = coef_df.head(15)
    top_neg = coef_df.tail(15)
    plot_df = pd.concat([top_pos, top_neg]).sort_values('Weight', ascending=True)
    
    # Color based on whether it drives engagement up or down.
    plot_df['Impact'] = np.where(plot_df['Weight'] > 0, 'Increases Comments', 'Decreases Comments')
    
    fig = px.bar(
        plot_df, x='Weight', y='Feature', color='Impact',
        title="GLM Nomogram Weights: What Drives Article Engagement?",
        labels={'Weight': 'Impact on Comments (Lasso Coefficient)'},
        color_discrete_map={'Increases Comments': '#2ecc71', 'Decreases Comments': '#e74c3c'},
        template=template_name
    )
    
    # Clean up the y-axis labels for readability.
    fig.update_yaxes(ticktext=[t.replace('Loc_', 'Location: ').replace('Topic_', 'Topic: ').replace('Sent_', 'Sentiment: ') for t in plot_df['Feature']], 
                     tickvals=plot_df['Feature'])   

    fig.update_layout(margin=dict(r=150))         
    fig.write_html(output_filename)
