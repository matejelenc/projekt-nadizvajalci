"""Code for preprocessing articles."""

from dataclasses import dataclass
from typing import Optional
import json

from geonews.nlp.util import get_id

@dataclass
class Article:
    """Maps to :Article label."""
    id: str
    title: Optional[str] = None
    url: Optional[str] = None
    figures: Optional[list[dict]] = None
    date: Optional[str] = None
    n_comments: Optional[int] = None

def preprocess(articles_path : str) -> list[Article]:
    """Extract relevant article information.

    Args:
        articles_path (str): Path to a jsonl file containing
            articles in the original format.

    Returns:
        list[Article]: List of Article instances.
    """
    articles : list[Article] = list()
    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            # Load article.
            if not line.strip(): continue
            record = json.loads(line)

            # Get id.
            article_id = get_id(record)
            if article_id == None: continue
            
            # Get number of comments.
            n_comments = record.get("n_comments", 0)
            if not n_comments: n_comments = 0
            else: n_comments = int(n_comments)

            article = Article(
                id=article_id,
                date=record.get("date", None),
                title=record.get("title", None),
                url=record.get("url", None),
                figures=record.get("figures", None),
                n_comments=n_comments,
            )
            articles.append(article)

    return articles

