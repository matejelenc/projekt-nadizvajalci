"Code for preprocessing article sentiment."

from dataclasses import dataclass
from typing import Generator
import json

from transformers import pipeline
from tqdm import tqdm

from geonews.nlp.util import get_id

@dataclass(frozen=True)
class Sentiment:
    """Maps to :Sentiment label."""
    id: str
    name: str

@dataclass(frozen=True)
class HasSentiment:
    # Article ID.
    id: str
    # Sentiment's ID.
    sentiment: str

def get_batch_generator(articles_path: str, batch_size: int) -> Generator[tuple[list[str], list[str]], None, None]:
    """Create a batch generator for preprocessing.
    
    Args:
        file_path (str): Path to jsonl file.
        batch_size (int): Batch size to use.
    
    Returns:
        Generator[tuple[list[str], list[str]], None, None]: Batch generator.
    """
    ids, contents = [], []
    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            # Load article.
            if not line.strip(): continue
            article = json.loads(line)

            # get article id.
            article_id = get_id(article)
            if article_id == None: continue
            ids.append(article_id)

            # Create content.
            title = article.get("title", "")
            lead = article.get("lead", "")

            content =  f"{title} {lead}"
            contents.append(content)

            if len(ids) == batch_size:
                yield (ids, contents)
                ids, contents = [], []

    if ids:
        yield (ids, contents)

class SentimentExtractor:
    """Pipeline for extracting sentiment from articles using zero shot classification model.
    
    Attributes:
        task (str): Task of the underlying model.
        model (str): Name of the transformers model.
        labels (list[str]): Labels to which the model classifies.
    """
    task: str = "zero-shot-classification"
    model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    labels: list[str] = ["dobra novica", "nevtralna novica", "slaba novica"]

    def __init__(self, cuda_id: int = -1):
        """Initialize.

        Args:
            cuda_id (int): ID for CUDA device on which to run the model.
        """
        self.pipe = pipeline(self.task, self.model, device=cuda_id) # pyright: ignore[reportCallIssue, reportArgumentType]
    
    def extract(self, contents: list[str]) -> list[str]:
        """
        Extract sentiment.

        Args:
            contents (list[str]): List of articles.
        
        Returns:
            list[str]: List of sentiments.
        """
        sentiments = self.pipe(contents, self.labels, multi_label=False)
        return [s["labels"][0] for s in sentiments]
    
    def __call__(self, contents: list[str]) -> list[str]:
        """Shorthand for extract."""
        return self.extract(contents)

def preprocess(
        articles_path : str,
        batch_size : int,
        extractor : SentimentExtractor,
        ) -> tuple[list[HasSentiment], list[Sentiment]]: 
    """Extract sentiment from articles.

    Assigns label 'dobra novica', 'slaba novica' or 'nevtralna novica' to 
    each article using sentiment ID.

    Args:
        articles_path (str): Path to a jsonl file containing articles. Each
            article should contain keys 'id' (or at least '_id'), 'title' and 
            'lead'. Since sentiment can usually be determined from this alone, 
            we do not use actual article content for efficiency.
        batch_size (int): Batch size for processing articles.
        extractor (SentimentExtractor): Zero-shot model for sentiment classification.
    
    Returns:
        tuple[list[Article], list[Sentiment]]: A list of Article instances containing
            article ID and sentiment ID, and a list Sentiment instances.

    """
    
    batch_generator = get_batch_generator(articles_path, batch_size)
    total_lines = sum(1 for _ in open(articles_path, "r", encoding="utf-8"))
    total_batches = (total_lines + batch_size - 1) // batch_size

    articles : list[HasSentiment] = list()
    sentiment_ids : dict[str, str] = {s: f"sentiment_{i}" for (i,s) in enumerate(SentimentExtractor.labels)}
    sentiments : list[Sentiment] = [Sentiment(id=sentiment_ids[s], name=s) for s in SentimentExtractor.labels]

    for (ids, contents) in tqdm(batch_generator, desc="Extracting...", total=total_batches):
        article_sentiments = extractor(contents)

        for id, sentiment in zip(ids, article_sentiments):
            article = HasSentiment(id=id, sentiment=sentiment_ids[sentiment])
            articles.append(article)
    
    return articles, sentiments