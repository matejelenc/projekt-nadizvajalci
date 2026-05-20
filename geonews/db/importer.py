"""Class for importing data to Neo4j."""

from typing import Optional
import json

from neo4j import GraphDatabase, Result

class Importer:
    """Class for importing jsonl records to Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize client.

        Args:
            uri (str): Database URI.
            user (str): Database user.
            password (str): Database password.

        """
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    def close(self):
        """Close client."""
        self.driver.close()

    def _query(self, cypher: str, parameters: Optional[dict] = None) -> Result:
        """Internal method for running queries.
        
        Args:
            cypher (str): Cypher query to run.
            parameters (Optional[dict]): Query parameters if any.
        
        Returns:
            Result: Result of query execution.
        
        """
        with self.driver.session() as session:
            return session.run(cypher, parameters) # type: ignore

    def _import_jsonl_batched(
        self,
        filepath: str,
        cypher_query: str,
        batch_size: int = 1000,
    ):
        """
        Import records from jsonl file in batches.

        Args:
            filepath (str): Path to jsonl file.
            cypher_query (str): Cypher query using UNWIND $rows AS row.
            batch_size (int): Batch size.
        """

        batch = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                # Remove keys with None values.
                cleaned = {
                    k: v for k, v in record.items()
                    if v is not None
                }
                batch.append(cleaned)

                if len(batch) >= batch_size:
                    self._query(cypher_query, {"rows": batch})
                    batch = []

            if batch:
                self._query(cypher_query, {"rows": batch})

    def import_sentiments(self, jsonl_path : str):
        """Import sentiments from jsonl file."""

        constraint = "CREATE CONSTRAINT sentiment_id IF NOT EXISTS FOR (s:Sentiment) REQUIRE s.id IS UNIQUE"
        self._query(constraint)

        query = """
        UNWIND $rows AS row

        MERGE (s:Sentiment {id: row.id})

        SET s.name = row.name
        """
        self._import_jsonl_batched(jsonl_path, query)

    def import_topics(self, jsonl_path : str):
        """Import topics from jsonl file."""

        constraint = "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE"
        self._query(constraint)

        query = """
        UNWIND $rows AS row

        MERGE (t:Topic {id: row.id})

        SET t.name = row.name
        """
        self._import_jsonl_batched(jsonl_path, query)

    def import_locations(self, jsonl_path : str):
        """Import locations from jsonl file."""

        constraint = "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE"
        self._query(constraint)

        query = """
        UNWIND $rows AS row

        MERGE (l:Location {id: row.id})

        SET
            l.name = row.name,
            l.latitude = row.latitude,
            l.longitude = row.longitude,
            l.type = row.type,
            l.admin_level = CASE WHEN row.admin_level IS NOT NULL THEN row.admin_level ELSE l.admin_level END

        WITH l, row
        CALL apoc.create.addLabels(l, [row.type]) YIELD node

        WITH l, row
        WHERE row.part_of IS NOT NULL AND row.part_of <> ""

        MERGE (p:Location {id: row.part_of})
        MERGE (l)-[:PART_OF]->(p)
        """
        self._import_jsonl_batched(jsonl_path, query)

    def import_articles(self, jsonl_path: str):
        """Import articles from jsonl file."""

        article_constraint = """
        CREATE CONSTRAINT article_id IF NOT EXISTS
        FOR (a:Article)
        REQUIRE a.id IS UNIQUE
        """

        figure_constraint = """
        CREATE CONSTRAINT figure_id IF NOT EXISTS
        FOR (f:Figure)
        REQUIRE f.id IS UNIQUE
        """

        self._query(article_constraint)
        self._query(figure_constraint)

        query = """
        UNWIND $rows AS row

        MERGE (a:Article {id: row.id})

        // Article properties
        FOREACH (_ IN CASE WHEN row.title IS NOT NULL THEN [1] ELSE [] END |
            SET a.title = row.title
        )

        FOREACH (_ IN CASE WHEN row.url IS NOT NULL THEN [1] ELSE [] END |
            SET a.url = row.url
        )

        FOREACH (_ IN CASE WHEN row.n_comments IS NOT NULL THEN [1] ELSE [] END |
            SET a.n_comments = row.n_comments
        )

        FOREACH (_ IN CASE WHEN row.date IS NOT NULL THEN [1] ELSE [] END |
            SET a.date = datetime(row.date)
        )

        // Figures
        FOREACH (fig IN coalesce(row.figures, []) |

            MERGE (f:Figure {
                id: row.id + "_" + coalesce(fig.img, "")
            })

            SET
                f.caption = fig.caption,
                f.img = fig.img,
                f.source = fig.source

            MERGE (a)-[:HAS_FIGURE]->(f)
        )
        """

        self._import_jsonl_batched(jsonl_path, query)


    def import_has_topic(self, jsonl_path: str):
        """Import has topic relations from jsonl file."""

        query = """
        UNWIND $rows AS row

        MATCH (a:Article {id: row.id})
        MATCH (t:Topic {id: row.topic})

        MERGE (a)-[:HAS_TOPIC]->(t)
        """
        self._import_jsonl_batched(jsonl_path, query)

    def import_has_sentiment(self, jsonl_path: str):
        """Import has sentiment relations from jsonl file."""

        query = """
        UNWIND $rows AS row

        MATCH (a:Article {id: row.id})
        MATCH (s:Sentiment {id: row.sentiment})

        MERGE (a)-[:HAS_SENTIMENT]->(s)
        """
        self._import_jsonl_batched(jsonl_path, query)

    def import_mentions_locations(self, jsonl_path: str):
        """Import mentioned locations relations from jsonl file."""

        query = """
        UNWIND $rows AS row

        MATCH (a:Article {id: row.id})

        UNWIND row.locations AS location_id

        MATCH (l:Location {id: location_id})

        MERGE (a)-[:MENTIONS_LOCATION]->(l)
        """
        self._import_jsonl_batched(jsonl_path, query)