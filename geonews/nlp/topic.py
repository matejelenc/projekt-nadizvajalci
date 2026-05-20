"""Code for preprocessing article topics."""

from dataclasses import dataclass
import json

from geonews.nlp.util import get_id

@dataclass(frozen=True)
class Topic:
    """Maps to :Topic label."""
    id: str
    name: str

@dataclass(frozen=True)
class HasTopic:
    # Article ID.
    id : str
    # Topic ID.
    topic: str

def preprocess(articles_path : str) -> tuple[list[HasTopic], list[Topic]]:
    """Extract relevant article information.

    Args:
        articles_path (str): Path to a jsonl file containing
            articles in the original format.

    Returns:
        tuple[list[Article], list[Topic]]: List of Article instances.
    """
    article_topics : list[dict[str, str]] = list()
    topic_names : set[str] = set()

    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            # Load article.
            if not line.strip(): continue
            record = json.loads(line)

            # Get id.
            article_id = get_id(record)
            if article_id == None: continue
            
            # Get topic.
            topic = record.get("topics", "unknown")

            article_topics.append({"id": article_id, "topic": topic})
            topic_names.add(topic)

    topic_ids = {t: f"topic_{i}" for i, t in enumerate(topic_names)}
    topics = [Topic(id=topic_ids[t], name=t) for t in topic_names]

    articles : list[HasTopic] = list()
    for record in article_topics:
        article = HasTopic(id=record["id"], topic=topic_ids[record["topic"]])
        articles.append(article)
    
    return articles, topics
