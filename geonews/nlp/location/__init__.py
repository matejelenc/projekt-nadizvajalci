"""Contains utility functions and classes for extracting locations from text.

- `extract`: Contains a pipeline for extracting possible location mentions from text.
- `cluster`: Contains a pipeline for clustering mentions into probable equal locations.
- `geo`: Contains a pipeline for matching mentions to an actual location and extracting
    location information - longitude, latitude, type etc.
- `preprocess`: Contains the entire pipeline.
"""

from .extract import LocationExtractor
from .cluster import cluster_location_mentions
from .geo import GeoNames, Location, Continent, Country, Region, City
from .pipeline import preprocess