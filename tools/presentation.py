"""Create some visualizations for a presentation."""
import os
import logging

from geonews.db import Client
from geonews import plot

logger = logging.getLogger(__name__)

def main():
    # Create client.
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_endpoint = os.getenv("DB_ENDPOINT")
    if not all([db_user, db_password, db_endpoint]):
        raise EnvironmentError("Copy .env.example to .env and fill in the values.")
    client = Client(db_endpoint, db_user, db_password) # type: ignore

    plot_dir = "./data/plots"
    os.makedirs(plot_dir, exist_ok=True)
    for location_type, scope_name in zip(["Region", "Country"], ["Slovenia", None]):
        for color_theme in ["dark", "light"]:
            # Run Louvain and plot on world map.
            output_path = os.path.join(plot_dir, f"louvain_{location_type}_{color_theme}.html")
            df = client.louvain(location_type, scope_name) # type: ignore
            plot.louvain(df, output_path, color_mode=color_theme) # type: ignore

            # Run UMAP and plot a scatter plot.
            output_path = os.path.join(plot_dir, f"umap_{location_type}_{color_theme}.html")
            df = client.umap(location_type, scope_name) # type: ignore
            plot.umap(df, output_path, color_mode=color_theme) # type: ignore

            # Run k-means and plot a scatter plot.
            output_path = os.path.join(plot_dir, f"kmeans_{location_type}_{color_theme}.html")
            df = client.kmeans(location_type, scope_name) # type: ignore
            plot.kmeans(df, output_path, "scatter", color_mode=color_theme) # type: ignore

            # Run enrichment analysis.
            output_path = os.path.join(plot_dir, f"enrichment_{location_type}_{color_theme}.html")
            df = client.enrichment(location_type, scope_name) # type: ignore
            plot.enrichment(df, output_path, color_mode=color_theme) # type: ignore

            # Run Lasso regression for engagement prediction.
            output_path = os.path.join(plot_dir, f"nomogram_{location_type}_{color_theme}.html")
            df = client.lasso_regression(location_type, scope_name) # type: ignore
            plot.nomogram(df, output_path, color_mode=color_theme) # type: ignore

            # Plot a Sankey diagram.
            output_path = os.path.join(plot_dir, f"sankey_{location_type}_{color_theme}.html")
            df = client.sankey(location_type, scope_name) # type: ignore
            plot.sankey(df, output_path, color_mode=color_theme) # type: ignore

            # Plot temporal evolution of topics.
            output_path = os.path.join(plot_dir, f"temporal_{location_type}_{color_theme}.html")
            df = client.temporal(location_type, scope_name) # type: ignore
            plot.temporal(df, output_path, color_mode=color_theme) # type: ignore 

    logger.info("Plots saved at: %s", plot_dir)     

if __name__ == "__main__":
    main()   