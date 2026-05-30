"""Classes for running Neo4j queries on graphs."""

from typing import Optional, Literal

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.cluster import KMeans
from scipy.stats import hypergeom
import numpy as np
import umap

from graphdatascience import GraphDataScience

class Client:
    """
    Client for performing advanced queries and algorithms on graphs.

    Attributes:
        gds (GraphDataScience): Neo4j Graph Data Science Python Client.
    """

    def __init__(self, endpoint: str, user : str, password : str):
        """Initialize.

        Args:
            endpoint (str): Endpoint.
            user (str): Database username.
            password (str): Database password.
        """
        self.gds = GraphDataScience(endpoint, auth=(user, password))

    def build_cooccurence_graph(self):
        """Builds a graph of location co-mentions.

        This should be ran only once, after importing data to the database.
        """

        cypher_query = """
        CALL apoc.periodic.iterate(
            "
            MATCH (a:Article)
            RETURN a
            ",

            "
            MATCH (a)-[:MENTIONS_LOCATION]->(l1:Location)
            MATCH (a)-[:MENTIONS_LOCATION]->(l2:Location)

            WHERE id(l1) < id(l2)

            MERGE (l1)-[r:CO_OCCURS_WITH]-(l2)

            ON CREATE SET r.weight = 1
            ON MATCH SET r.weight = r.weight + 1
            ",

            {
                batchSize: 1000,
                parallel: true
            }
        )
        """
        self.gds.run_cypher(cypher_query)

    def _ml_features(
            self, 
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None
        ) -> pd.DataFrame:
        """
        Fetches a tabular dataset of article mentions for specific location types, 
        optionally filtered by a parent geographic scope.

        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world for which to query. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
        
        Returns:
            pd.DataFrame: The fetched table.
        """
        
        if scope_name:
            match_location = f"""
            MATCH (l:Location {{type: '{location_type}'}})-[:PART_OF*0..5]->(parent:Location {{name: '{scope_name}'}})
            """
        else:
            match_location = f"MATCH (l:Location {{type: '{location_type}'}})"

        query = f"""
            {match_location}
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l)
            OPTIONAL MATCH (a)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
            RETURN l.name AS name, l.latitude AS lat, l.longitude AS lon,
                a.id AS article_id, a.n_comments AS comments, 
                t.name AS topic, s.name AS sentiment
        """
        df = self.gds.run_cypher(query)
        if df.empty: return df

        # Clean and enforce numeric values.
        df = df.dropna(subset=['lat', 'lon', 'name'])
        for col in ['lat', 'lon']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['lat', 'lon'])
        return df

    def sankey(
            self, 
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None
        ) -> pd.DataFrame:
        """Fetch data for a sankey diagram."""
        return self._ml_features(location_type, scope_name)

    def umap(
            self,
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None,
            min_articles:int = 1
        ) -> pd.DataFrame:
        """
        Calculates topic/sentiment distributions per location and reduces them to 2D space.

        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world on which to run UMAP. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
            min_articles (int): Minimum number of articles for considering a location.

        Return:
            pd.DataFrame: Table with UMAP coordinates.
        """
        # Fetch data.
        df = self._ml_features(location_type, scope_name)

        # Filter out locations with too little data.
        article_counts = df.groupby('name')['article_id'].nunique()
        valid_locations = article_counts[article_counts >= min_articles].index
        df_filtered = df[df['name'].isin(valid_locations)]
        
        # Create probability distributions.
        topic_dist = pd.crosstab(df_filtered['name'], df_filtered['topic'], normalize='index')
        sentiment_dist = pd.crosstab(df_filtered['name'], df_filtered['sentiment'], normalize='index')
        
        # Combine features.
        feature_matrix = pd.concat([topic_dist, sentiment_dist], axis=1)
        
        # Reduce dimensionality.
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Run PCA to reduce noise.
        n_components = min(10, scaled_features.shape[1])
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)
        
        # Project to 2D using UMAP.
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(pca_features)
        
        # Construct final table.
        results = pd.DataFrame({
            'name': feature_matrix.index,
            'umap_x': embedding[:, 0], # type: ignore
            'umap_y': embedding[:, 1], # type: ignore
            'article_count': article_counts[feature_matrix.index].values
        })
        
        # Add back coordinaes.
        coords = df_filtered[['name', 'lat', 'lon']].drop_duplicates(subset=['name'])
        return results.merge(coords, on='name', how='left')
    
    def kmeans(
            self,
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None, 
            k_clusters: int = 4, 
            min_articles: int = 1
        ) -> pd.DataFrame:
        """
        Groups locations based on Engagement (comments) and Sentiment.

        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world on which to run k-means. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
            k_clusters (int): Number of clusters.
            min_articles (int): Minimum number of articles required to consider a location.
        
        Returns:
            pd.DataFrame: Locations with cluster ids.
        """ 

        # Fetch data.
        df = self._ml_features(location_type, scope_name)

        # Map sentiments to numerical values for averaging.
        sent_map = {'dobra novica': 1, 'nevtralna novica': 0, 'slaba novica': -1}
        df['sent_score'] = df['sentiment'].map(sent_map).fillna(0)
        
        metrics = df.groupby('name').agg(
            avg_comments=('comments', 'mean'),
            avg_sentiment=('sent_score', 'mean'),
            article_count=('article_id', 'nunique'),
            lat=('lat', 'first'),
            lon=('lon', 'first')
        ).reset_index()
        
        # Filter noise.
        metrics = metrics[metrics['article_count'] >= min_articles]
        
        # Cluster.
        features_to_cluster = metrics[['avg_comments', 'avg_sentiment']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_to_cluster)
        
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
        metrics['cluster_id'] = kmeans.fit_predict(scaled_features)
        metrics['cluster_id'] = metrics['cluster_id'].astype(str) # For categorical coloring
        
        return metrics

    def louvain(
            self,
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None,
            min_articles: int = 1,
            tolerance: float = 0.0001
        ) -> pd.DataFrame:
    
        """
        Runs Louvain on a part of the world (specific country, continent).
        Uses .stream() so no permanent changes are made to the database.
        
        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world on which to run Louvain. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
        
        Returns:
            (pd.DataFrame): Table containing data about edges - weight, nodes' info and community id,
                with columns: 'source', 'source_lat', 'source_lon', 'target', 'target_lat', 'target_lon', 
                'weight', 'community'.
        """
        # Create node query.
        if scope_name:
            node_query = f"""
            MATCH (l:Location {{type: '{location_type}'}})-[:PART_OF*0..5]->(parent:Location {{name: '{scope_name}'}})
            WITH l
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l)
            WITH l, count(DISTINCT a) as article_count
            WHERE article_count >= {min_articles}
            RETURN id(l) AS id
            """
        else:
            node_query = f"""
            MATCH (l:Location {{type: '{location_type}'}})
            WITH l
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l)
            WITH l, count(DISTINCT a) as article_count
            WHERE article_count >= {min_articles}
            RETURN id(l) AS id
            """

        # Create edge query.
        edge_query = f"""
        MATCH (s:Location {{type: '{location_type}'}})-[r:CO_OCCURS_WITH]-(t:Location {{type: '{location_type}'}})
        RETURN id(s) AS source, id(t) AS target, r.weight AS weight
        """

        # Project graph.
        if self.gds.graph.exists("scoped_graph")["exists"]:
            self.gds.graph.drop(self.gds.graph.get("scoped_graph"))
            
        # Handle dangling edges.
        G, _ = self.gds.graph.project.cypher(
            "scoped_graph", 
            node_query, 
            edge_query,
            validateRelationships=False 
        )

        # Run Louvain and stream to python.
        communities = self.gds.louvain.stream(
            G, 
            relationshipWeightProperty="weight",
            tolerance=tolerance
        )
        communities = communities.drop(columns=["intermediateCommunityIds"])
        communities.columns = ["nodeId", "community"]

        # Fetch edges and join with community data.
        results = self.gds.run_cypher(f"""
            MATCH (l1:Location {{type: '{location_type}'}})-[r:CO_OCCURS_WITH]-(l2:Location {{type: '{location_type}'}})
            RETURN id(l1) as sourceId, l1.name as source, 
                l1.latitude as source_lat, l1.longitude as source_lon, 
                id(l2) as targetId, l2.name as target, 
                l2.latitude as target_lat, l2.longitude as target_lon, 
                r.weight as weight
        """)

        # Merge community info onto dedges.
        df = results.merge(communities, left_on="sourceId", right_on="nodeId").rename(columns={"community": "source_community"})
        df = df.merge(communities, left_on="targetId", right_on="nodeId").rename(columns={"community": "target_community"})
        
        # Filter to only keep edges within the same community.
        df = df[df["source_community"] == df["target_community"]]
        df = df.rename(columns={"source_community": "community"})

        G.drop()
        return df[["source", "source_lat", "source_lon", "target", "target_lat", "target_lon", "weight", "community"]]
    
    def temporal(
            self,
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None
        ) -> pd.DataFrame:
        """
        Extract temporal data for time-series analysis.
        
        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world for which to extract temporal data. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
        
        Returns:
            (pd.DataFrame): Fetched table.
        """

        # Create query.
        match_location = f"MATCH (l:Location {{type: '{location_type}'}})-[:PART_OF*0..5]->(parent:Location {{name: '{scope_name}'}})" if scope_name else f"MATCH (l:Location {{type: '{location_type}'}})"
        query = f"""
            {match_location}
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l)
            MATCH (a)-[:HAS_TOPIC]->(t:Topic)
            RETURN a.id AS article_id, toString(a.date) AS publish_date, t.name AS topic
        """

        # Run query.
        df = self.gds.run_cypher(query)
        if df.empty: return df
            
        # Parse datetime.
        df['publish_date'] = df['publish_date'].astype(str).str[:19]
        df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        
        df = df.dropna(subset=['publish_date'])
        return df
    
    def lasso_regression(
            self,
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None
        ) -> pd.DataFrame:
        """
        Trains a Lasso regression model to predict comments and extracts the feature weights.

        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world on which to run regression. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
        
        Returns:
            pd.DataFrame: Table with features and coefficients.
        """

        # Fetch data.
        df = self._ml_features(location_type, scope_name)

        # Clean data and handle skewness in comments (using log1p).
        df_model = df.dropna(subset=['topic', 'sentiment', 'comments']).copy()
        df_model['target_comments'] = np.log1p(df_model['comments']) 
        
        # One-hot encode the categorical variables.
        X = pd.get_dummies(df_model[['name', 'topic', 'sentiment']], prefix=['Loc', 'Topic', 'Sent'])
        y = df_model['target_comments']
        
        # Standardize the features so Lasso penalizes them equally.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Lasso with cross-validation to automatically find the best alpha (regularization penalty).
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Extract coefficients.
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Weight': lasso.coef_
        })
        
        # Filter out features that Lasso reduced to absolute zero (the "noise").
        coef_df = coef_df[coef_df['Weight'] != 0].sort_values('Weight', ascending=False)
        return coef_df
    
    def enrichment(
            self,
            location_type: str = "Settlement", 
            scope_name: Optional[str] = None,
            min_articles: int = 5
        ) -> pd.DataFrame:
        """
        Calculates hypergeometric p-values to find statistically enriched topics per location.

        Args:
            location_type (str): Location type to use as nodes.
            scope_name (str): Part of the world on which to run Louvain. An example would
                location_type='Settlement' and scope_name='Slovenia', or location_type='SovereignState'
                and scope_name='Europe'.
            min_articles (int): Minimum number of articles to consider a location.
        """
        # Fetch data.
        df = self._ml_features(location_type, scope_name)

        # Total articles in the whole dataset (Population size: M).
        M = df['article_id'].nunique()
        
        # Total articles for each topic globally (Total successes in population: n).
        topic_totals = df.groupby('topic')['article_id'].nunique().to_dict()
        
        # Total articles for each location (Sample size: N).
        location_totals = df.groupby('name')['article_id'].nunique()
        
        enrichment_data = []
        
        # Group by Location and Topic to get local successes (k)
        grouped = df.groupby(['name', 'topic'])['article_id'].nunique().reset_index()
        grouped.rename(columns={'article_id': 'k'}, inplace=True)
        
        for _, row in grouped.iterrows():
            loc = row['name']
            topic = row['topic']
            k = row['k'] # Mentions of topic in this location.
            N = location_totals[loc] # Total mentions of this location.
            n = topic_totals.get(topic, 0) # Total mentions of this topic globally.
            
            if N < min_articles or pd.isna(topic):
                continue
                
            p_val = hypergeom.sf(k - 1, M, n, N)
            neg_log_p = -np.log10(p_val + 1e-300)
            
            enrichment_data.append({
                'name': loc,
                'topic': topic,
                'k': k,
                'p_value': p_val,
                'significance': neg_log_p
            })
            
        df_enrich = pd.DataFrame(enrichment_data)
        
        # Filter for standard statistical significance (p < 0.05).
        df_enrich = df_enrich[df_enrich["p_value"] < 0.05].sort_values("significance", ascending=False)
        
        coords = df[["name", "lat", "lon"]].drop_duplicates(subset=["name"])
        return df_enrich.merge(coords, on="name", how="left")
    
    def articles_for_location(self, location_name: str) -> pd.DataFrame:
        """
        Fetches all articles mentioning a specific location.
        
        Args:
            location_name (str): Name of the location.
        
        Returns:
            pd.DataFrame: Table with article data.
        """
        query = """
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l:Location {name: $name})
            OPTIONAL MATCH (a)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
            RETURN a.id AS id, a.title AS title, a.url AS url, 
                a.img AS img, a.date AS date, a.n_comments AS n_comments,
                t.name AS topic, s.name AS sentiment
            ORDER BY a.date DESC
        """
        df = self.gds.run_cypher(query, params={"name": location_name})
        if df.empty:
            return df
        df['date'] = df['date'].astype(str).str[:19]
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        return df
    
    def search_locations(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Search locations by alias."""
        q = query.lower().strip()
        result = self.gds.run_cypher("""
            MATCH (a:LocationAlias)-[:ALIAS_OF]->(l:Location)
            WHERE toLower(a.value) CONTAINS $q
            WITH l, a
            ORDER BY size(a.value)
            WITH l, collect(a.value)[0] AS matched_alias
            RETURN l.name AS name, l.type AS type,
                l.latitude AS lat, l.longitude AS lon,
                matched_alias
            LIMIT $top_k
        """, params={"q": q, "top_k": top_k})
        return result
    
    def cluster_keywords(self, location_names: list, top_k: int = 20) -> pd.DataFrame:
        """
        Gets top keywords for a cluster of locations using TF-IDF style comparison.
        Counts keywords in cluster articles vs all articles.

        Args:
            location_names (list): List of location names in the cluster.
            top_k (int): Number of top keywords to return.

        Returns:
            pd.DataFrame: Table with keywords and their relative frequency.
        """
        # Get keyword counts for articles mentioning cluster locations
        cluster_query = """
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l:Location)
            WHERE l.name IN $locations
            MATCH (a)-[:HAS_KEYWORD]->(k)
            RETURN k.normalized AS keyword, count(DISTINCT a) AS cluster_count
        """
        df_cluster = self.gds.run_cypher(cluster_query, params={"locations": location_names})
        if df_cluster.empty:
            return df_cluster

        # Get total keyword counts across all articles
        total_query = """
            MATCH (a:Article)-[:HAS_KEYWORD]->(k)
            RETURN k.normalized AS keyword, count(DISTINCT a) AS total_count
        """
        df_total = self.gds.run_cypher(total_query)

        # Merge and calculate relative frequency
        df = df_cluster.merge(df_total, on="keyword", how="left")
        total_articles = self.gds.run_cypher("MATCH (a:Article) RETURN count(a) AS total").iloc[0]["total"]
        cluster_articles = self.gds.run_cypher("""
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l:Location)
            WHERE l.name IN $locations
            RETURN count(DISTINCT a) AS total
        """, params={"locations": location_names}).iloc[0]["total"]

        # TF-IDF style score: how much more common in cluster vs globally
        df["cluster_freq"] = df["cluster_count"] / max(cluster_articles, 1)
        df["global_freq"] = df["total_count"] / max(total_articles, 1)
        df["score"] = df["cluster_freq"] / (df["global_freq"] + 0.001)

        # Filter short/meaningless keywords
        df = df[df["keyword"].str.len() > 2]
        df = df[df["cluster_count"] >= 2]

        return df.nlargest(top_k, "score")[["keyword", "cluster_count", "score"]]


    def cluster_sentiment(self, location_names: list) -> pd.DataFrame:
        """
        Gets sentiment distribution for a cluster vs global average.

        Args:
            location_names (list): List of location names in the cluster.

        Returns:
            pd.DataFrame: Table with sentiment counts for cluster and globally.
        """
        # Sentiment for cluster
        cluster_query = """
            MATCH (a:Article)-[:MENTIONS_LOCATION]->(l:Location)
            WHERE l.name IN $locations
            MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
            RETURN s.name AS sentiment, count(DISTINCT a) AS cluster_count
        """
        df_cluster = self.gds.run_cypher(cluster_query, params={"locations": location_names})

        # Global sentiment
        global_query = """
            MATCH (a:Article)-[:HAS_SENTIMENT]->(s:Sentiment)
            RETURN s.name AS sentiment, count(DISTINCT a) AS global_count
        """
        df_global = self.gds.run_cypher(global_query)

        df = df_cluster.merge(df_global, on="sentiment", how="outer").fillna(0)

        # Normalize to percentages
        df["cluster_pct"] = df["cluster_count"] / df["cluster_count"].sum() * 100
        df["global_pct"] = df["global_count"] / df["global_count"].sum() * 100

        return df
    
    def get_location_center(self, location_name: str) -> tuple:
        """
        Gets the center coordinates for a location by name.
        
        Args:
            location_name (str): Name of the location.
        
        Returns:
            tuple: (lat, lon) or None if not found.
        """
        result = self.gds.run_cypher("""
            MATCH (l:Location {name: $name})
            RETURN l.latitude AS lat, l.longitude AS lon
            LIMIT 1
        """, params={"name": location_name})
        
        if result.empty or pd.isna(result.iloc[0]["lat"]):
            return None
        return (float(result.iloc[0]["lat"]), float(result.iloc[0]["lon"]))