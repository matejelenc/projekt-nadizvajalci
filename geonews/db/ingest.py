"""Script for exporting preprocessed data to Neo4j database."""

import logging
import argparse
import os
from dotenv import load_dotenv

from geonews.db import Importer, Client

logger = logging.getLogger(__name__)

def setup_logging():
    """Sets up a simple logger."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_args():
    """Parses arguments for importing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", 
        nargs='*', 
        type=str,
        default=["article", "topic", "sentiment", "location", "graph"],
        help="Which data to import, subset of [article, topic, sentiment, location, graph]."
    )
    return parser.parse_args()

def main():
    # Create importer.
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_endpoint = os.getenv("DB_ENDPOINT")

    if db_user is None or db_password is None or db_endpoint is None:
        logger.error("Database configuration not provided in .env, cannot connect.")
    importer = Importer(db_endpoint, db_user, db_password) # type: ignore

    clean_dir = "./data/clean"
    args = get_args()
    pipeline = args.pipeline

    # Import articles.
    if "article" in pipeline:
        article_path = os.path.join(clean_dir, "articles.jsonl")
        importer.import_articles(article_path)
        logger.info("Imported articles.")

    # Import topics.
    if "topic" in pipeline:
        topics_path = os.path.join(clean_dir, "topics.jsonl")
        importer.import_topics(topics_path)
        logger.info("Imported topics.")

        has_topic_path = os.path.join(clean_dir, "has_topic.jsonl")
        importer.import_has_topic(has_topic_path)
        logger.info("Imported article->topic relations.")

    # Import sentiments.
    if "sentiment" in pipeline:
        sentiments_path = os.path.join(clean_dir, "sentiments.jsonl")
        importer.import_sentiments(sentiments_path)
        logger.info("Imported sentiments.")

        has_sent_path = os.path.join(clean_dir, "has_sentiment.jsonl")
        importer.import_has_sentiment(has_sent_path)
        logger.info("Imported article->sentiment relations.")

    # Import locations.
    if "location" in pipeline:
        locations_path = os.path.join(clean_dir, "locations.jsonl")
        importer.import_locations(locations_path)

        mentions_loc_path = os.path.join(clean_dir, "mentions_locations.jsonl")
        importer.import_mentions_locations(mentions_loc_path)
        logger.info("Imported article->locations relations.")

    # Build co-occurence graph.
    if "graph" in pipeline:
        client = Client(db_endpoint, db_user, db_password) # type: ignore
        client.build_cooccurence_graph()
        logger.info("Build co-occurence graph.")

if __name__ == "__main__":
    main()