"""Pipeline for location mention extraction, clustering and mapping to actual locations."""

from __future__ import annotations
from typing import Mapping, Generator
from dataclasses import dataclass, field
import json

from tqdm import tqdm

from geonews.nlp.location import GeoNames, Location, LocationExtractor, cluster_location_mentions
from geonews.nlp.util import get_id

@dataclass(frozen=True)
class MentionsLocations:
    # Article ID.
    id : str

    # Mentioned locations' IDs.
    locations: list[str] = field(default_factory=list)

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

            paragraphs = article.get("paragraphs", [])
            if isinstance(paragraphs, list):
                paragraphs = "\n".join(paragraphs)

            content =  f"{title} {lead} {paragraphs}"
            contents.append(content)

            if len(ids) == batch_size:
                yield (ids, contents)
                ids, contents = [], []

    if ids:
        yield (ids, contents)

def mentions_to_locations(
    mention_counts: Mapping[str, int],
    geonames: GeoNames,
    *,
    similarity: float = 0.90,
    min_count: int = 1,
    allow_risky_containment: bool = False,
    frequency_boost_strength: float = 0.03
) -> tuple[list[Location], dict[str, str]]:
    """
    Execute the full automatic pipeline for location extraction and resolution.

    The pipeline performs the following steps:
    1.  Clusters raw lemmatized mentions based on string similarity.
    2.  Resolves clusters using the GeoNames database.
    3.  Generates standardized Location objects (City, Region, Country, etc.).
    4.  Maps every input mention to a specific resolved Location ID.

    Args:
        mention_counts (Mapping[str, int]): Mapping from raw lemmatized mention 
            strings to their global frequency in the corpus.
        geonames (GeoNames): An initialized resolver instance for 
            GeoNames database lookups and ranking.
        similarity (float): The base similarity threshold for the initial 
            clustering phase. Defaults to 0.88.
        min_count (int): Frequency threshold; mentions appearing fewer times 
             than this are ignored. Defaults to 1.
        allow_risky_containment (bool): If False, prevents merging mentions where 
            one is a modified version of another (e.g., "Korea" and "North Korea"). 
            Defaults to False.
        frequency_boost_strength (float): The strength of the similarity boost 
            applied when two mentions both have high frequency. Defaults to 0.02.

    Returns:
        tuple[list[Location], dict[str, str]]: A tuple containing:
            - A list of all unique resolved Location objects, sorted by ID.
            - A dictionary mapping each raw lemmatized mention string to its 
              corresponding resolved Location ID.
    """
    clusters = cluster_location_mentions(
        mention_counts,
        similarity=similarity,
        min_count=min_count,
        allow_risky_containment=allow_risky_containment,
        frequency_boost_strength=frequency_boost_strength,
    )

    mention_to_location_id: dict[str, str] = {}

    for cluster in clusters:
        cluster_mentions = sorted(
            cluster.mentions,
            key=lambda m: mention_counts.get(m, 0),
            reverse=True,
        )

        cluster_locations = geonames.extract(
            queries=cluster_mentions,
            mention_counts=mention_counts,
        )
        for mention in cluster_mentions:
            single_locations = geonames.extract(
                queries=[mention],
                mention_counts=mention_counts,
            )

            if len(single_locations) == 1:
                loc = single_locations[0]
                mention_to_location_id[mention] = loc.id
                continue

            if len(single_locations) > 1:
                loc = single_locations[0]
                mention_to_location_id[mention] = loc.id
                continue

            if len(cluster_locations) == 1:
                loc = cluster_locations[0]
                mention_to_location_id[mention] = loc.id
                continue

    locations = geonames.get_all_locations()
    locations = sorted(locations, key=lambda loc: loc.id)

    return locations, mention_to_location_id

def preprocess(
        articles_path : str,  
        batch_size: int,
        extractor: LocationExtractor,
        geonames: GeoNames,
        similarity: float = 0.90,
        min_count: int = 1,
        allow_risky_containment: bool = False,
        frequency_boost_strength: float = 0.03
        ) -> tuple[list[MentionsLocations], list[Location]]:
    """Extracts locations from articles.

    First, the pipeline extracts all possible location mentions
    from articles, which are then clustered into probable equal
    locations and then mapped to Location instances using the
    GeoNames database. Lastly, location mentions are mapped to
    actual location IDs.

    Args:
        articles_path (str): Path to a jsonl file containing
            articles. Each record should have keys 'id' (or at
            least '_id'), 'title', 'lead' and 'paragraphs',
            following the format of the original yaml file.
        batch_size (int): Batch size at which to process articles.
        extractor (LocatioExtractor): Model for performing NER.
        geonames (GeoNames): GeoNames client for mapping mentions
            to real locations.
        similarity (float): Similarity threshold for clustering
            mentions into probable equal locations.
        min_count (int): Minimum number of mentions for a location
            to be considered valid.
    
    Returns:
        tuple[list[Article], list[Location]]: List of Article objects
            which contain the articles ID and mentioned locations' IDs,
            and a list of Location objects, containing all directly or
            indirectly mentioned locations. For example, if city 'Kranj'
            is mentioned, region 'Gorenjska', country 'Slovenia' and
            continent 'Europe' are considered indirect mentions.
    """
    
    batch_generator = get_batch_generator(articles_path, batch_size)
    total_lines = sum(1 for _ in open(articles_path, "r", encoding="utf-8"))
    total_batches = (total_lines + batch_size - 1) // batch_size


    location_counts : dict[str, int] = dict()
    article_mentions : list[dict] = list()
    for (ids, contents) in tqdm(batch_generator, desc="Extracting...", total=total_batches):
        location_mentions, location_counts = extractor(contents, location_counts)

        for id, mentions in zip(ids, location_mentions):
            article = {"id": id, "locations": mentions}
            article_mentions.append(article)

    locations, location_to_id = mentions_to_locations(
        mention_counts=location_counts, 
        geonames=geonames, 
        similarity=similarity, 
        min_count=min_count,
        allow_risky_containment=allow_risky_containment,
        frequency_boost_strength=frequency_boost_strength
    )

    articles : list[MentionsLocations] = list()
    for record in article_mentions:
            location_ids : set[str] = set()
            for alias in record["locations"]:
                if alias in location_to_id:
                    location_ids.add(location_to_id[alias])
            ids = list(location_ids)

            article = MentionsLocations(id=record["id"], locations=ids)
            articles.append(article)

    return articles, locations