"""Code for mapping raw location mentions to actual locations using GeoNames."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Optional, Iterable
import re
import unicodedata

import pandas as pd

@dataclass(frozen=True)
class Location:
    """Base class. Maps to the generic :Location label in Neo4j."""
    id: str
    name: str
    latitude: float
    longitude: float

    @property
    def labels(self) -> list[str]:
        """Helper to generate Neo4j labels dynamically (e.g., ['Location', 'City'])."""
        return ["Location", self.__class__.__name__]

@dataclass(frozen=True)
class City(Location):
    """Maps to the :City label."""
    # Replaces region/country/continent strings.
    # Stores the ID of the immediate parent to build the [:PART_OF] edge.
    part_of: Optional[str] = None 

@dataclass(frozen=True)
class Region(Location):
    """Maps to the :Region label."""
    part_of: Optional[str] = None 

@dataclass(frozen=True)
class Country(Location):
    """Maps to the :Country label."""
    part_of: Optional[str] = None 

@dataclass(frozen=True)
class Continent(Location):
    """Maps to the :Continent label. Top of the hierarchy."""
    pass

@dataclass(frozen=True)
class GeoCandidate:
    """
    Candidate GeoNames match for a location mention.

    Attributes:
        geonameid (int): GeoNames identifier of the candidate location.
        matched_query (str): Original query or mention that produced the match.
        matched_name (str): GeoNames name matched for the candidate.
        score (float): Ranking score assigned to the candidate.
        support_count (int): Number of mentions supporting the candidate.
    """
    geonameid: int
    matched_query: str
    matched_name: str
    score: float
    support_count: int


class GeoNames:
    """
    Resolve extracted location mentions to GeoNames-backed Location objects.

    The resolver builds fast lookup indices over GeoNames data, ranks candidate
    locations conservatively, and creates hierarchical Location objects for
    continents, countries, regions, and cities.

    Important behavior:
        - A soft cluster may resolve to several Location objects.
        - Ambiguous names are not collapsed during indexing.
        - Resolution is automatic and conservative.

    Attributes:
        CITY_FEATURE_CLASS (str): GeoNames feature class used for populated
            places.
        REGION_FEATURE_CODES (set[str]): GeoNames feature codes treated as
            administrative regions.
        COUNTRY_FEATURE_CODES (set[str]): GeoNames feature codes treated as
            countries.
        CONTINENT_FEATURE (tuple[str, str]): GeoNames feature class and code
            identifying continents.
        EUROPE_CODE (str): Continent code used for Europe in countryInfo.txt.
        country_map (dict): Country metadata keyed by ISO country code.
        region_map (dict): Region names keyed by full region code.
        df (pd.DataFrame): Filtered GeoNames data indexed by geonameid.
        country_coords (dict): Country coordinates keyed by country code.
        region_coords (dict): Region coordinates keyed by full region code.
        continent_coords (dict): Continent coordinates keyed by normalized name.
        resolved_locations (dict[str, Location]): Created Location objects keyed
            by internal location ID.
        name_to_ids (dict[str, list[int]]): Normalized names mapped to matching
            GeoNames IDs.
    """

    # Cities, towns, villages, populated places.
    CITY_FEATURE_CLASS = "P"

    # Administrative regions.
    REGION_FEATURE_CODES = {
        "ADM1",
        "ADM2",
        "ADM3",
        "ADM4",
    }

    # Countries.
    COUNTRY_FEATURE_CODES = {
        "PCLI",
    }

    # Continents.
    CONTINENT_FEATURE = ("L", "CONT")

    # Europe continent code in countryInfo.txt.
    EUROPE_CODE = "EU"

    def __init__(
        self,
        all_countries_path: str,
        country_info_path: str,
        admin1_path: str,
    ):
        """
        Initialize a GeoNames location resolver.

        Args:
            all_countries_path (str): Path to the GeoNames allCountries.txt
                dataset.
            country_info_path (str): Path to the GeoNames countryInfo.txt
                dataset.
            admin1_path (str): Path to the GeoNames admin1CodesASCII.txt
                dataset.
        """
        self.country_map, self.region_map = self._load_mappings(
            country_info_path,
            admin1_path,
        )

        self.df = self._load_all_countries(all_countries_path)

        self.country_coords = (
            self.df[self.df["feature_code"] == "PCLI"]
            .set_index("country_code")[["latitude", "longitude"]]
            .to_dict("index")
        )

        self.region_coords = (
            self.df[self.df["feature_code"] == "ADM1"]
            .set_index("region_code_full")[["latitude", "longitude"]]
            .to_dict("index")
        )

        self.continent_coords = (
            self.df[
                (self.df["feature_class"] == "L")
                & (self.df["feature_code"] == "CONT")
            ]
            .set_index("name_norm")[["latitude", "longitude"]]
            .to_dict("index")
        )

        self.resolved_locations: dict[str, Location] = {}

        self._build_fast_indices()

    @staticmethod
    def _load_mappings(
        country_info_path: str,
        admin1_path: str,
    ) -> tuple[dict, dict]:
        """
        Load country and administrative-region metadata.

        Args:
            country_info_path (str): Path to the GeoNames countryInfo.txt file.
            admin1_path (str): Path to the GeoNames admin1CodesASCII.txt file.

        Returns:
            tuple[dict, dict]: Country metadata keyed by ISO country code and
                administrative region names keyed by full region code.
        """
        countries_df = pd.read_csv(
            country_info_path,
            sep="\t",
            comment="#",
            header=None,
            keep_default_na=False,
            usecols=[0, 4, 8],
            names=["iso", "country_name", "continent_code"],
        )

        continent_map = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "OC": "Oceania",
            "SA": "South America",
            "AN": "Antarctica",
        }

        countries_df["continent_name"] = countries_df["continent_code"].map(
            continent_map
        )

        country_dict = countries_df.set_index("iso").to_dict("index")

        admin1_df = pd.read_csv(
            admin1_path,
            sep="\t",
            header=None,
            keep_default_na=False,
            usecols=[0, 1],
            names=["code", "region_name"],
        )

        region_dict = admin1_df.set_index("code")["region_name"].to_dict()

        return country_dict, region_dict

    @classmethod
    def _load_all_countries(cls, path: str) -> pd.DataFrame:
        """
        Load and filter the GeoNames allCountries dataset.

        Args:
            path (str): Path to the GeoNames allCountries.txt dataset.

        Returns:
            pd.DataFrame: Filtered GeoNames records containing populated places,
                administrative regions, countries, and continents, with
                normalized name fields and region codes added.
        """
        columns = [
            "geonameid",
            "name",
            "asciiname",
            "alternatenames",
            "latitude",
            "longitude",
            "feature_class",
            "feature_code",
            "country_code",
            "cc2",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "population",
            "elevation",
            "dem",
            "timezone",
            "modification_date",
        ]

        use_cols = [
            "geonameid",
            "name",
            "asciiname",
            "alternatenames",
            "latitude",
            "longitude",
            "feature_class",
            "feature_code",
            "country_code",
            "admin1_code",
            "population",
        ]

        df = pd.read_csv(
            path,
            sep="\t",
            names=columns,
            usecols=use_cols,
            keep_default_na=False,
            low_memory=False,
        )

        df = df[
            (
                (df["feature_class"] == "P")
                | (
                    (df["feature_class"] == "A")
                    & (
                        df["feature_code"].isin(
                            cls.REGION_FEATURE_CODES | cls.COUNTRY_FEATURE_CODES
                        )
                    )
                )
                | (
                    (df["feature_class"] == "L")
                    & (df["feature_code"] == "CONT")
                )
            )
        ].copy()

        df["population"] = pd.to_numeric(
            df["population"],
            errors="coerce",
        ).fillna(0)

        df["geonameid"] = pd.to_numeric(
            df["geonameid"],
            errors="coerce",
        ).astype(int)

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").fillna(0.0)
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").fillna(0.0)

        df["name_norm"] = df["name"].map(cls.normalize_name)
        df["ascii_norm"] = df["asciiname"].map(cls.normalize_name)
        df["region_code_full"] = df["country_code"] + "." + df["admin1_code"]

        return df

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize a location name for matching.

        This keeps Slovene diacritics by default. If data inconsistently drops
        č/š/ž, an additional ASCII-folded index can be used.

        Args:
            name (str): Location name to normalize.

        Returns:
            str: Lowercased, Unicode-normalized, punctuation-stripped name.
        """
        if name is None:
            return ""

        name = str(name).strip().lower()
        name = unicodedata.normalize("NFC", name)
        name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)
        name = re.sub(r"_", " ", name)
        name = re.sub(r"\s+", " ", name).strip()

        return name

    @staticmethod
    def ascii_fold(name: str) -> str:
        """
        Remove combining diacritical marks from a string.

        Args:
            name (str): Input string to fold to ASCII-like form.

        Returns:
            str: String with combining diacritical marks removed.
        """
        name = unicodedata.normalize("NFD", name)
        return "".join(ch for ch in name if unicodedata.category(ch) != "Mn")

    def _build_fast_indices(self):
        """
        Build normalized-name lookup indices: name_to_ids[name] -> list[geonameid].
        That is necessary for splitting soft clusters.
        """
        name_to_ids: dict[str, set[int]] = defaultdict(set)

        for row in self.df.itertuples(index=False):
            gid = int(row.geonameid) # type: ignore

            names = {
                self.normalize_name(row.name), # type: ignore
                self.normalize_name(row.asciiname), # type: ignore
            }

            if pd.notna(row.alternatenames) and row.alternatenames:
                for alt in str(row.alternatenames).split(","):
                    names.add(self.normalize_name(alt))

            for name in names:
                if name:
                    name_to_ids[name].add(gid)

                    folded = self.ascii_fold(name)
                    if folded and folded != name:
                        name_to_ids[folded].add(gid)

        self.name_to_ids: dict[str, list[int]] = {
            name: sorted(ids)
            for name, ids in name_to_ids.items()
        }

        self.df.set_index("geonameid", inplace=True, drop=False)

    def _is_city(self, row: pd.Series) -> bool:
        """
        Check whether a GeoNames row represents a populated place.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            bool: True if the row is a city or populated place.
        """
        return row["feature_class"] == self.CITY_FEATURE_CLASS

    def _is_region(self, row: pd.Series) -> bool:
        """
        Check whether a GeoNames row represents an administrative region.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            bool: True if the row is an administrative region.
        """
        return (
            row["feature_class"] == "A"
            and row["feature_code"] in self.REGION_FEATURE_CODES
        )

    def _is_country(self, row: pd.Series) -> bool:
        """
        Check whether a GeoNames row represents a country.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            bool: True if the row is a country.
        """
        return (
            row["feature_class"] == "A"
            and row["feature_code"] in self.COUNTRY_FEATURE_CODES
        )

    def _is_continent(self, row: pd.Series) -> bool:
        """
        Check whether a GeoNames row represents a continent.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            bool: True if the row is a continent.
        """
        return (
            row["feature_class"] == self.CONTINENT_FEATURE[0]
            and row["feature_code"] == self.CONTINENT_FEATURE[1]
        )

    def _is_valid_location_type(self, row: pd.Series) -> bool:
        """
        Check whether a GeoNames row is a supported location type.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            bool: True if the row is a city, region, country, or continent.
        """
        return (
            self._is_city(row)
            or self._is_region(row)
            or self._is_country(row)
            or self._is_continent(row)
        )

    def _country_info(self, country_code: str) -> dict:
        """
        Get country metadata for an ISO country code.

        Args:
            country_code (str): ISO country code.

        Returns:
            dict: Country metadata, or an empty dictionary if unavailable.
        """
        return self.country_map.get(country_code, {})

    def _is_european(self, row: pd.Series) -> bool:
        """
        Check whether a GeoNames row belongs to a European country.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            bool: True if the row belongs to a country marked as European.
        """
        country_code = row.get("country_code", "")
        info = self._country_info(country_code)
        return info.get("continent_code") == self.EUROPE_CODE

    def _type_rank(self, row: pd.Series) -> int:
        """
        Assign a type-based ranking score to a GeoNames row.

        Higher is better.

        Priority requested:
            Slovenian locations
            continents
            countries
            European cities
            other cities by population

        Regions are inserted after countries, but Slovenian regions are already
        boosted by the Slovenian-location priority.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            int: Type rank used as part of candidate scoring.
        """
        if self._is_continent(row):
            return 90

        if self._is_country(row):
            return 80

        if self._is_region(row):
            return 70

        if self._is_city(row):
            if self._is_european(row):
                return 60
            return 50

        return 0

    def _candidate_score(
        self,
        row: pd.Series,
        query: str,
        support_count: int,
    ) -> float:
        """
        Produce a ranking score for a GeoNames candidate.

        The absolute value is not meaningful; only relative ordering matters.

        Args:
            row (pd.Series): GeoNames row for the candidate.
            query (str): Original mention query.
            support_count (int): Number of mentions supporting this query.

        Returns:
            float: Candidate ranking score, or -1.0 for unsupported location
                types.
        """
        if not self._is_valid_location_type(row):
            return -1.0

        query_norm = self.normalize_name(query)
        name_norm = row["name_norm"]
        ascii_norm = row["ascii_norm"]

        exact_primary = query_norm == name_norm
        exact_ascii = query_norm == ascii_norm
        exact_name_bonus = 30 if exact_primary else 15 if exact_ascii else 0

        is_slovenian = row["country_code"] == "SI"
        slovenia_bonus = 10_000 if is_slovenian else 0

        type_bonus = self._type_rank(row) * 100

        population = float(row.get("population", 0) or 0)
        population_bonus = min(1000.0, population ** 0.5)

        support_bonus = min(500.0, support_count ** 0.5 * 25.0)

        return (
            slovenia_bonus
            + type_bonus
            + exact_name_bonus
            + population_bonus
            + support_bonus
        )

    def _get_or_create_continent(self, continent_name: str) -> Optional[str]:
        """
        Get or create a continent Location object.

        Args:
            continent_name (str): Name of the continent.

        Returns:
            Optional[str]: Internal continent location ID, or None if the
                continent name is missing or unknown.
        """
        if not continent_name or continent_name == "unknown":
            return None

        continent_id = f"continent_{continent_name.replace(' ', '_').lower()}"

        if continent_id not in self.resolved_locations:
            coords = self.continent_coords.get(
                self.normalize_name(continent_name),
                {"latitude": 0.0, "longitude": 0.0},
            )

            self.resolved_locations[continent_id] = Continent(
                id=continent_id,
                name=continent_name,
                latitude=float(coords["latitude"]),
                longitude=float(coords["longitude"]),
            )

        return continent_id

    def _get_or_create_country(
        self,
        c_code: str,
        country_name: str,
        continent_name: str,
    ) -> Optional[str]:
        """
        Get or create a country Location object.

        Args:
            c_code (str): ISO country code.
            country_name (str): Country name.
            continent_name (str): Parent continent name.

        Returns:
            Optional[str]: Internal country location ID, or None if the country
                code is missing or unknown.
        """
        if not c_code or c_code == "unknown":
            return None

        parent_id = self._get_or_create_continent(continent_name)
        country_id = f"country_{c_code}"

        if country_id not in self.resolved_locations:
            coords = self.country_coords.get(
                c_code,
                {"latitude": 0.0, "longitude": 0.0},
            )

            self.resolved_locations[country_id] = Country(
                id=country_id,
                name=country_name,
                latitude=float(coords["latitude"]),
                longitude=float(coords["longitude"]),
                part_of=parent_id,
            )

        return country_id

    def _get_or_create_region(
        self,
        region_code_full: str,
        region_name: str,
        c_code: str,
        country_name: str,
        continent_name: str,
    ) -> Optional[str]:
        """
        Get or create an administrative-region Location object.

        Args:
            region_code_full (str): Full region code in the form
                ``country.admin1``.
            region_name (str): Region name.
            c_code (str): ISO country code.
            country_name (str): Country name.
            continent_name (str): Parent continent name.

        Returns:
            Optional[str]: Internal region location ID, or the parent country ID
                when the region code is unavailable.
        """
        if (
            not region_code_full
            or region_code_full == "unknown"
            or region_code_full.endswith(".")
        ):
            return self._get_or_create_country(c_code, country_name, continent_name)

        parent_id = self._get_or_create_country(
            c_code,
            country_name,
            continent_name,
        )

        region_id = f"region_{region_code_full}"

        if region_id not in self.resolved_locations:
            coords = self.region_coords.get(
                region_code_full,
                {"latitude": 0.0, "longitude": 0.0},
            )

            self.resolved_locations[region_id] = Region(
                id=region_id,
                name=region_name,
                latitude=float(coords["latitude"]),
                longitude=float(coords["longitude"]),
                part_of=parent_id,
            )

        return region_id

    def _candidate_ids_for_query(self, query: str) -> list[int]:
        """
        Find GeoNames candidate IDs for a normalized query.

        Args:
            query (str): Location mention query.

        Returns:
            list[int]: Candidate GeoNames IDs matching the query or its
                ASCII-folded form.
        """
        q_norm = self.normalize_name(query)

        if not q_norm:
            return []

        ids = set(self.name_to_ids.get(q_norm, []))

        folded = self.ascii_fold(q_norm)
        if folded != q_norm:
            ids.update(self.name_to_ids.get(folded, []))

        return list(ids)

    def _best_candidate_for_query(
        self,
        query: str,
        support_count: int,
    ) -> Optional[GeoCandidate]:
        """
        Select the highest-scoring GeoNames candidate for a query.

        Args:
            query (str): Location mention query.
            support_count (int): Number of mentions supporting the query.

        Returns:
            Optional[GeoCandidate]: Best candidate for the query, or None if no
                valid location candidate is found.
        """
        candidate_ids = self._candidate_ids_for_query(query)

        if not candidate_ids:
            return None

        best: Optional[GeoCandidate] = None

        for gid in candidate_ids:
            row = self.df.loc[gid]

            if not self._is_valid_location_type(row): # type: ignore
                continue

            score = self._candidate_score(row, query, support_count) # type: ignore

            candidate = GeoCandidate(
                geonameid=int(gid),
                matched_query=query,
                matched_name=str(row["name"]),
                score=score,
                support_count=support_count,
            )

            if best is None or candidate.score > best.score:
                best = candidate

        return best

    def _collect_cluster_candidates(
        self,
        queries: Iterable[str],
        mention_counts: Optional[Mapping[str, int]] = None,
    ) -> dict[int, GeoCandidate]:
        """
        Resolve cluster queries independently and group candidates by GeoNames ID.

        This is the key step that allows a soft cluster to split.

        Args:
            queries (Iterable[str]): Location mentions belonging to one soft
                cluster.
            mention_counts (Optional[Mapping[str, int]]): Optional global
                mention counts used as support values.

        Returns:
            dict[int, GeoCandidate]: Best grouped candidate for each GeoNames ID.
        """
        grouped: dict[int, GeoCandidate] = {}

        for query in queries:
            query = str(query).strip()
            if not query:
                continue

            support_count = 1
            if mention_counts is not None:
                support_count = int(mention_counts.get(query, 1))

            best = self._best_candidate_for_query(query, support_count)

            if best is None:
                continue

            existing = grouped.get(best.geonameid)

            if existing is None:
                grouped[best.geonameid] = best
            else:
                grouped[best.geonameid] = GeoCandidate(
                    geonameid=best.geonameid,
                    matched_query=existing.matched_query
                    if existing.support_count >= best.support_count
                    else best.matched_query,
                    matched_name=best.matched_name,
                    score=max(existing.score, best.score),
                    support_count=existing.support_count + best.support_count,
                )

        return grouped

    def _should_keep_split_candidate(
        self,
        candidate: GeoCandidate,
        all_candidates: list[GeoCandidate],
    ) -> bool:
        """
        Decide whether a split candidate should be retained.

        This is intentionally permissive for exact independent evidence: if a
        mention in the cluster directly resolves to a valid distinct location,
        keep it.

        The main filtering is already done by:
            - exact name/alternate-name lookup
            - valid type restrictions
            - GeoNames ranking

        Args:
            candidate (GeoCandidate): Candidate under consideration.
            all_candidates (list[GeoCandidate]): All candidates generated for
                the same soft cluster.

        Returns:
            bool: True if the candidate should survive cluster splitting.
        """
        if not all_candidates:
            return False

        if len(all_candidates) == 1:
            return True

        # Keep all strongly supported candidates.
        if candidate.support_count >= 2:
            return True

        # Keep singleton candidates if their score is close to the best.
        best_score = max(c.score for c in all_candidates)
        return candidate.score >= best_score * 0.80

    def _location_id_for_row(self, row: pd.Series) -> Optional[str]:
        """
        Build an internal Location ID for a GeoNames row.

        Args:
            row (pd.Series): GeoNames row.

        Returns:
            Optional[str]: Internal location ID, or None if the row type is not
                supported.
        """
        gid = int(row["geonameid"])
        f_class = row["feature_class"]
        f_code = row["feature_code"]
        c_code = row["country_code"]
        a1_code = row["admin1_code"]
        region_code_full = f"{c_code}.{a1_code}"

        if f_class == "P":
            return f"city_{gid}"

        if f_class == "A" and f_code in self.COUNTRY_FEATURE_CODES:
            return f"country_{c_code}"

        if f_class == "A" and f_code in self.REGION_FEATURE_CODES:
            return f"region_{region_code_full}"

        if f_class == "L" and f_code == "CONT":
            cont_name = str(row["name"]).replace(" ", "_").lower()
            return f"continent_{cont_name}"

        return None

    def _candidate_to_location(self, candidate: GeoCandidate) -> Optional[Location]:
        """
        Convert a GeoNames candidate into a Location object.

        Creates any required parent hierarchy objects and caches the resulting
        Location in resolved_locations.

        Args:
            candidate (GeoCandidate): Candidate to convert.

        Returns:
            Optional[Location]: Created or cached Location object, or None if
                the candidate cannot be converted.
        """
        row = self.df.loc[candidate.geonameid]

        if not self._is_valid_location_type(row): # type: ignore
            return None

        loc_id = self._location_id_for_row(row) # type: ignore
        if loc_id is None:
            return None

        if loc_id in self.resolved_locations:
            return self.resolved_locations[loc_id]

        lat = float(row["latitude"]) # type: ignore
        lon = float(row["longitude"]) # type: ignore
        f_class = row["feature_class"]
        f_code = row["feature_code"]
        c_code = row["country_code"]
        a1_code = row["admin1_code"]

        country_info = self.country_map.get(c_code, {})
        country_name = country_info.get("country_name", "unknown")
        continent_name = country_info.get("continent_name", "unknown")

        region_code_full = f"{c_code}.{a1_code}"
        region_name = self.region_map.get(region_code_full, "unknown")


        display_name = str(row["name"])
        if f_class == "P": # type: ignore
            parent_id = self._get_or_create_region(
                region_code_full,
                region_name,
                c_code, # type: ignore
                country_name,
                continent_name,
            )

            loc = City(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
                part_of=parent_id,
            )

        elif f_class == "A" and f_code in self.COUNTRY_FEATURE_CODES: # type: ignore
            parent_id = self._get_or_create_continent(continent_name)

            loc = Country(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
                part_of=parent_id,
            )

        elif f_class == "A" and f_code in self.REGION_FEATURE_CODES: # type: ignore
            parent_id = self._get_or_create_country(
                c_code, # type: ignore
                country_name,
                continent_name,
            )

            loc = Region(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
                part_of=parent_id,
            )

        elif f_class == "L" and f_code == "CONT": # type: ignore
            loc = Continent(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
            )

        else:
            return None

        self.resolved_locations[loc_id] = loc
        return loc

    def extract(
        self,
        queries: list[str],
        mention_counts: Optional[Mapping[str, int]] = None,
    ) -> list[Location]:
        """
        Resolve a soft cluster of mentions into zero, one, or several locations.

        Args:
            queries (list[str]): Mentions belonging to one soft cluster.
            mention_counts (Optional[Mapping[str, int]]): Optional global
                mention counts. Used as support when splitting a soft cluster.

        Returns:
            list[Location]: Empty list if nothing valid was found, one item for
                normal clusters, or multiple items if the soft cluster contains
                several real places.
        """
        if not queries:
            return []

        grouped = self._collect_cluster_candidates(
            queries,
            mention_counts=mention_counts,
        )

        if not grouped:
            return []

        candidates = sorted(
            grouped.values(),
            key=lambda c: (c.score, c.support_count),
            reverse=True,
        )

        kept_candidates = [
            c for c in candidates
            if self._should_keep_split_candidate(c, candidates)
        ]

        locations: list[Location] = []

        seen_ids: set[str] = set()

        for candidate in kept_candidates:
            loc = self._candidate_to_location(candidate)

            if loc is None:
                continue

            if loc.id in seen_ids:
                continue

            seen_ids.add(loc.id)
            locations.append(loc)

        return locations

    def get_all_locations(self) -> list[Location]:
        """
        Return all resolved Location objects created by the resolver.

        Returns:
            list[Location]: Cached resolved locations, including hierarchy
                locations created as parents.
        """
        return list(self.resolved_locations.values())