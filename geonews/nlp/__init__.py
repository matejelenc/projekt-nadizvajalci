"""
Contains code for NLP.

- 'location': Code for extracting location mentions, clustering them into groups likely
    representing the same location and mapping them to actual physical locations.
- 'sentiment': Code for classifying article's sentiment.
- 'topic': Code for extracting article topics.
- 'article': Code for extracting relevant article info.
- 'util': Helper functions.
- 'preprocess': Script for preprocessing the MMC dataset.
"""

from . import location, sentiment, topic, article, util