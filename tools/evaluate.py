import os
import yaml
import json

def yaml_to_jsonl(yaml_path: str, jsonl_path: str):
    """Convert yaml file to jsonl.

    Args:
        yaml_path (str): Path to yaml file.
        jsonl_path (str): Path where to save.
    """
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    with open(jsonl_path, "w", encoding="utf-8") as file:
        for article in data:
            file.write(json.dumps(article, ensure_ascii=False) + '\n')

def to_json(data: list, jsonl_path: str):
    """Save list to json.

    Args:
        data (list): The data.
        jsonl_path (str): Where to save.
    """
    with open(jsonl_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def read_jsonl(file_path: str) -> list[dict]:
    """Read jsonl file.

    Args:
        file_path (str): Path to jsonl file.
    
    Returns:
        list[dict]: Records.
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            records.append(json.loads(line))
    return records


def append_to_jsonl(data: list[dict], file_path: str):
    """Add records to jsonl file.
    
    Args:
        data (list[dict]): Records.
        file_path (str): Path to jsonl file.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

annotations = read_jsonl("./data/evaluate/annotations.jsonl")
articles = read_jsonl("./data/evaluate/articles.jsonl")

test = []
for article in articles:
    for annotation in annotations:
        if article["id"] == annotation["id"]:
            record = {
                "id": article["id"],
                "sentiment": article["sentiment"],
                "true_sentiment": annotation["sentiment"],
                "locations": article["locations"],
                "true_locations": annotation["locations"]
            }
            test.append(record)
            break
append_to_jsonl(test, "./data/evaluate/test.jsonl")

