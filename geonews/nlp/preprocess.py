"""Preprocess the MMC data."""

import os
import logging
import argparse

from dataclasses import asdict

from geonews.nlp.util import yaml_to_jsonl, to_jsonl

logger = logging.getLogger(__name__)

def setup_logging():
    """Sets up a simple logger."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def get_args():
    """Parses arguments for preprocessing."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline", 
        nargs='*', 
        type=str,
        default=["location"],
        help="Parts of preprocessing to execute."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for processing articles on CUDA."
    )
    parser.add_argument(
        "--cuda_id", 
        type=int, 
        default=0, 
        help="ID of the CUDA device to use."
    )

    parser.add_argument(
        "--similarity", 
        type=float, 
        default=0.95, 
        help="Threshold for location similarity."
    )

    parser.add_argument(
        "--min_count", 
        type=int, 
        default=1, 
        help="Minimum location count."
    )

    parser.add_argument(
        "--frequency_boost_strength", 
        type=float, 
        default=0.03, 
    )

    parser.add_argument(
        "--allow_risky_containment", 
        type=bool, 
        default=False, 
    )

    return parser.parse_args()

def process_articles(jsonl_path : str, clean_dir : str):
    """Helper function for extracting relevant article info."""
    from geonews.nlp import article

    articles = article.preprocess(jsonl_path)
    logger.info("Preprocessed articles.")

    article_data = [asdict(a) for a in articles]
    article_path = os.path.join(clean_dir, "articles.jsonl")
    to_jsonl(article_data, article_path)
    logger.info("Articles saved at: %s", article_path)

def process_topics(jsonl_path : str, clean_dir : str):
    """Helper function for extracting topics."""
    from geonews.nlp import topic

    has_topic, topics = topic.preprocess(jsonl_path)
    logger.info("Preprocessed topics.")

    topic_data = [asdict(t) for t in topics]
    topic_path = os.path.join(clean_dir, "topics.jsonl")
    to_jsonl(topic_data, topic_path)
    logger.info("Topics saved at: %s", topic_path)

    has_topic_data = [asdict(t) for t in has_topic]
    has_topic_path = os.path.join(clean_dir, "has_topic.jsonl")
    to_jsonl(has_topic_data, has_topic_path)
    logger.info("Article->Topic mapping saved at: %s", has_topic_path)

def process_sentiments(jsonl_path: str, clean_dir : str, cuda_id : int, batch_size : int):
    """Helper function for extracting sentiments."""
    from geonews.nlp import sentiment

    sent_extractor = sentiment.SentimentExtractor(cuda_id)
    has_sentiment, sentiments = sentiment.preprocess(jsonl_path, batch_size, sent_extractor)
    logger.info("Preprocessed sentiments.")

    sentiment_data = [asdict(s) for s in sentiments]
    sentiment_path = os.path.join(clean_dir, "sentiments.jsonl")
    to_jsonl(sentiment_data, sentiment_path)
    logger.info("Sentiments saved at: %s", sentiment_path)

    has_sentiment_data = [asdict(s) for s in has_sentiment]
    has_sentiment_path = os.path.join(clean_dir, "has_sentiment.jsonl")
    to_jsonl(has_sentiment_data, has_sentiment_path)
    logger.info("Article->Sentiment mapping saved at: %s", has_sentiment_path)

def process_locations(
        jsonl_path : str, 
        clean_dir : str, 
        raw_dir : str, 
        cuda_id : int,
        batch_size : int,
        similarity : float,
        min_count : int,
        allow_risky_containment : bool,
        frequency_boost_strength : float
    ):
    """Helper function for extracting locations."""
    from geonews.nlp import location

    all_countries_path = os.path.join(raw_dir, "allCountries.txt")
    country_info_path = os.path.join(raw_dir, "countryInfo.txt")
    admin1_path = os.path.join(raw_dir, "admin1CodesASCII.txt")

    loc_extractor = location.LocationExtractor(cuda_id)
    geonames = location.GeoNames(all_countries_path, country_info_path, admin1_path)
    mentions_locations, locations = location.preprocess(
        articles_path=jsonl_path, 
        batch_size=batch_size,
        extractor=loc_extractor,
        geonames=geonames,
        similarity=similarity,
        min_count=min_count,
        allow_risky_containment=allow_risky_containment,
        frequency_boost_strength=frequency_boost_strength
    )
    logger.info("Preprocessed locations.")

    location_data = [{**asdict(l), "type": l.__class__.__name__} for l in locations]
    location_path = os.path.join(clean_dir, "locations.jsonl")
    to_jsonl(location_data, location_path)
    logger.info("Locations saved at: %s", location_path)

    mentions_loc_data = [asdict(l) for l in mentions_locations]
    mention_loc_path = os.path.join(clean_dir, "mentions_locations.jsonl")
    to_jsonl(mentions_loc_data, mention_loc_path)
    logger.info("Article->Location mapping saved at %s", mention_loc_path)

def main():
    # Setup logging and parse arguments.
    setup_logging()
    args = get_args()

    # Create directory for database ready data.
    root_dir = "./data"
    raw_dir = os.path.join(root_dir, "raw")
    clean_dir = os.path.join(root_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    # Convert yaml to jsonl for faster loading.
    yaml_path = os.path.join(raw_dir, "mmc.yaml")
    jsonl_path = os.path.join(raw_dir, "mmc.jsonl")

    if not os.path.exists(jsonl_path):
        yaml_to_jsonl(yaml_path, jsonl_path)
    logger.info("Moved data from yaml -> jsonl.")

    
    # Execute preprocessing pipeline.
    if "article" in args.pipeline:
        process_articles(jsonl_path, clean_dir)

    if "topic" in args.pipeline:
        process_topics(jsonl_path, clean_dir)

    if "sentiment" in args.pipeline:
        process_sentiments(
            jsonl_path=jsonl_path, 
            clean_dir=clean_dir, 
            cuda_id=args.cuda_id, 
            batch_size=args.batch_size
        )
    
    if "location" in args.pipeline:
        process_locations(
            jsonl_path=jsonl_path,
            clean_dir=clean_dir,
            raw_dir=raw_dir,
            cuda_id=args.cuda_id,
            batch_size=args.batch_size,
            similarity=args.similarity,
            min_count=args.min_count,
            allow_risky_containment=args.allow_risky_containment,
            frequency_boost_strength=args.frequency_boost_strength
        )

if __name__ == "__main__":
    main()