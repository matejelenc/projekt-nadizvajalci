"""Resolve raw location mentions to GeoNames-backed locations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional
import re
import unicodedata

import pandas as pd


@dataclass(frozen=True)
class Location:
    """Base location node. Always maps to :Location plus its concrete label.
    
    Attributes:
        id (str): The unique identifier for the location.
        name (str): The canonical name of the location.
        latitude (float): The latitude coordinate.
        longitude (float): The longitude coordinate.
    """
    id: str
    name: str
    latitude: float
    longitude: float

    @property
    def labels(self) -> list[str]:
        """Retrieves the hierarchical graph labels for the location node.
        
        Returns:
            list[str]: A list of labels, starting with "Location" followed by 
                the concrete class name.
        """
        return ["Location", self.__class__.__name__]


@dataclass(frozen=True)
class Settlement(Location):
    """A populated place: city, town, village, locality, etc.
    
    Attributes:
        part_of (Optional[str]): The ID of the parent administrative division 
            or sovereign state this settlement belongs to.
    """
    part_of: Optional[str] = None


@dataclass(frozen=True)
class AdministrativeDivision(Location):
    """A subnational or country-like administrative unit.
    
    Attributes:
        part_of (Optional[str]): The ID of the parent sovereign state or 
            higher-level administrative division.
        admin_level (Optional[int]): The administrative level (e.g., 1 for ADM1).
    """
    part_of: Optional[str] = None
    admin_level: Optional[int] = None


@dataclass(frozen=True)
class SovereignState(Location):
    """A sovereign-state-level location.
    
    Attributes:
        part_of (Optional[str]): The ID of the parent continent.
    """
    part_of: Optional[str] = None


@dataclass(frozen=True)
class Continent(Location):
    """A continent-level location."""
    pass


# Backward-compatible aliases for older imports. New objects still use the new
# class names, so Location.labels returns Settlement / AdministrativeDivision /
# SovereignState.
City = Settlement
Region = AdministrativeDivision
Country = SovereignState


@dataclass(frozen=True)
class GeoCandidate:
    """A candidate matched from GeoNames for a given location query.
    
    Attributes:
        geonameid (int): The unique GeoNames ID for the matched candidate.
        matched_query (str): The original raw query string that produced this match.
        matched_name (str): The canonical name of the matched location.
        score (float): The confidence or similarity score of the match.
        support_count (int): The global mention count supporting this candidate.
    """
    geonameid: int
    matched_query: str
    matched_name: str
    score: float
    support_count: int


class GeoNames:
    """
    Resolve extracted location mentions against GeoNames.

    Operational hierarchy used by the resolver:

        Settlement -> AdministrativeDivision -> SovereignState -> Continent

    This hierarchy is intended for downstream filtering and graph analysis, not
    for strict legal/geopolitical classification. For example, municipalities,
    provinces, states, statistical regions, and special administrative regions
    are all represented as AdministrativeDivision nodes.
    """

    # GeoNames feature class P = populated places.
    SETTLEMENT_FEATURE_CLASS = "P"

    # GeoNames ADM levels. ADM1 is usually the most useful filter level, but the
    # lower levels are kept when available so settlements can attach to their
    # most specific known administrative parent.
    ADMINISTRATIVE_DIVISION_FEATURE_CODES = {"ADM1", "ADM2", "ADM3", "ADM4"}

    # Region-like non-administrative GeoNames features. They are still mapped
    # to AdministrativeDivision in this operational hierarchy so traditional,
    # statistical, and economic regions can be queried alongside ADM1-ADM4.
    REGIONLIKE_FEATURE_CODES = {"RGN", "RGNE"}

    # Country-like political entities that should not normally be exposed as
    # sovereign states.
    NON_SOVEREIGN_POLITICAL_FEATURE_CODES = {
        "PCLD",   # dependent political entity
        "PCLIX",  # section of independent political entity
        "PCLS",   # semi-independent political entity
        "TERR",   # territory
    }

    # GeoNames' independent political entity feature code. This is used as the
    # default sovereign-state proxy, optionally constrained by an explicit
    # sovereign_state_country_codes allowlist.
    SOVEREIGN_STATE_FEATURE_CODES = {"PCLI"}

    # Country-like ISO codes that should be parented to a sovereign state.
    DEFAULT_SOVEREIGN_STATE_OVERRIDES = {
        "HK": "CN",  # Hong Kong -> China
        "MO": "CN",  # Macau -> China
    }

    CONTINENT_FEATURE = ("L", "CONT")
    EUROPE_CODE = "EU"

    def __init__(
        self,
        all_countries_path: str,
        country_info_path: str,
        admin1_path: str,
        sovereign_state_country_codes: Optional[Iterable[str]] = None,
        sovereign_state_overrides: Optional[Mapping[str, str]] = None,
    ):
        """Initializes the GeoNames resolver with required data files.

        Args:
            all_countries_path (str): Path to the GeoNames allCountries.txt file.
            country_info_path (str): Path to the GeoNames countryInfo.txt file.
            admin1_path (str): Path to the GeoNames admin1CodesASCII.txt file.
            sovereign_state_country_codes (Optional[Iterable[str]]): An optional
                allowlist of ISO country codes to treat strictly as sovereign states.
            sovereign_state_overrides (Optional[Mapping[str, str]]): An optional
                mapping of ISO country codes to override their parent sovereign state.
        """
        self.sovereign_state_overrides = dict(
            self.DEFAULT_SOVEREIGN_STATE_OVERRIDES
        )
        if sovereign_state_overrides is not None:
            self.sovereign_state_overrides.update(
                {
                    str(child).upper(): str(parent).upper()
                    for child, parent in sovereign_state_overrides.items()
                    if child and parent
                }
            )

        self.country_map, self.region_map = self._load_mappings(
            country_info_path,
            admin1_path,
        )
        self.df = self._load_all_countries(all_countries_path)

        if sovereign_state_country_codes is None:
            self.sovereign_state_country_codes = (
                self._derive_sovereign_state_country_codes()
            )
        else:
            self.sovereign_state_country_codes = {
                str(code).upper()
                for code in sovereign_state_country_codes
                if str(code).strip()
            }

        self.sovereign_state_coords = (
            self.df[self.df["feature_code"].isin(self.SOVEREIGN_STATE_FEATURE_CODES)]
            .set_index("country_code")[["latitude", "longitude"]]
            .to_dict("index")
        )

        self.admin1_coords = (
            self.df[self.df["feature_code"] == "ADM1"]
            .set_index("region_code_full")[["latitude", "longitude"]]
            .to_dict("index")
        )

        self.administrative_division_index = (
            self._build_administrative_division_index()
        )

        self.continent_coords = (
            self.df[
                (self.df["feature_class"] == self.CONTINENT_FEATURE[0])
                & (self.df["feature_code"] == self.CONTINENT_FEATURE[1])
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
        """Loads and processes country and administrative division region mappings.

        Args:
            country_info_path (str): Path to the GeoNames countryInfo.txt file.
            admin1_path (str): Path to the GeoNames admin1CodesASCII.txt file.

        Returns:
            tuple[dict, dict]: A tuple containing two mapping dictionaries:
                - country_dict (dict): Maps ISO country codes to country info 
                  dictionaries (containing iso, country_name, continent_code, 
                  and continent_name).
                - region_dict (dict): Maps admin1 full region codes to region names.
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
        """Loads and filters the primary GeoNames dataset from a file path.

        Args:
            path (str): Path to the tab-delimited GeoNames text file 
                (e.g., allCountries.txt).

        Returns:
            pd.DataFrame: A processed DataFrame containing relevant location 
                columns, parsed numerical types, and normalized name variants.
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
            "admin2_code",
            "admin3_code",
            "admin4_code",
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
                (df["feature_class"] == cls.SETTLEMENT_FEATURE_CLASS)
                | (
                    (df["feature_class"] == "A")
                    & df["feature_code"].isin(
                        cls.ADMINISTRATIVE_DIVISION_FEATURE_CODES
                        | cls.NON_SOVEREIGN_POLITICAL_FEATURE_CODES
                        | cls.SOVEREIGN_STATE_FEATURE_CODES
                    )
                )
                | (
                    (df["feature_class"] == cls.CONTINENT_FEATURE[0])
                    & (
                        df["feature_code"].isin(
                            cls.REGIONLIKE_FEATURE_CODES
                            | {cls.CONTINENT_FEATURE[1]}
                        )
                    )
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
        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        ).fillna(0.0)
        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        ).fillna(0.0)

        for code_col in [
            "country_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
        ]:
            df[code_col] = df[code_col].astype(str).str.strip()

        df["name_norm"] = df["name"].map(cls.normalize_name)
        df["ascii_norm"] = df["asciiname"].map(cls.normalize_name)
        df["region_code_full"] = df["country_code"] + "." + df["admin1_code"]

        return df

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalizes a location name string for consistent textual matching.

        Lowercases, collapses whitespace, transforms non-word characters into spaces, 
        and ensures uniform Unicode normalization.

        Args:
            name (str): The raw location name string to normalize.

        Returns:
            str: The cleaned, lowercased, and sanitized string.
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
        """Folds non-ASCII Unicode strings into standard base ASCII characters.

        Removes diacritics and accents (e.g., converting 'São Paulo' to 'Sao Paulo').

        Args:
            name (str): The text string containing accented characters.

        Returns:
            str: The stripped string containing only the base character equivalents.
        """
        name = unicodedata.normalize("NFD", name)
        return "".join(ch for ch in name if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _clean_code(value: object) -> str:
        """Sanitizes and normalizes an administrative or country code value.

        Args:
            value (object): The raw code object (string, float, or None) to clean.

        Returns:
            str: A clean, whitespace-stripped string, or an empty string if the 
                input represents an invalid or missing value.
        """
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    @classmethod
    def _admin_level_from_feature_code(cls, feature_code: object) -> Optional[int]:
        """Extracts the numerical level from a standard GeoNames administrative code.

        Args:
            feature_code (object): The feature code string (e.g., 'ADM1', 'ADM2').

        Returns:
            Optional[int]: The integer level (1 to 4) if a valid match is found, 
                otherwise None.
        """
        match = re.fullmatch(r"ADM([1-4])", str(feature_code))
        if not match:
            return None
        return int(match.group(1))

    @classmethod
    def _admin_code_from_parts(
        cls,
        country_code: object,
        admin1_code: object = "",
        admin2_code: object = "",
        admin3_code: object = "",
        admin4_code: object = "",
        level: Optional[int] = None,
    ) -> Optional[str]:
        """Constructs a fully qualified dot-separated administrative division string.

        Args:
            country_code (object): The parent country ISO code.
            admin1_code (object, optional): The level 1 admin code. Defaults to "".
            admin2_code (object, optional): The level 2 admin code. Defaults to "".
            admin3_code (object, optional): The level 3 admin code. Defaults to "".
            admin4_code (object, optional): The level 4 admin code. Defaults to "".
            level (Optional[int], optional): Explict target depth level to build. 
                If None, the depth is automatically calculated based on the 
                most specific non-empty code provided.

        Returns:
            Optional[str]: A dot-separated administrative path path string 
                (e.g., 'US.CA.037') or None if inputs are non-contiguous or invalid.
        """
        country = cls._clean_code(country_code).upper()
        admin_codes = [
            cls._clean_code(admin1_code),
            cls._clean_code(admin2_code),
            cls._clean_code(admin3_code),
            cls._clean_code(admin4_code),
        ]

        if not country:
            return None

        if level is None:
            level = 0
            for idx, code in enumerate(admin_codes, start=1):
                if code:
                    level = idx
                elif any(admin_codes[idx:]):
                    # Non-contiguous admin codes are unsafe to use.
                    return None
            if level == 0:
                return None

        if level < 1 or level > 4:
            return None

        required_codes = admin_codes[:level]
        if len(required_codes) != level or any(not code for code in required_codes):
            return None

        return ".".join([country] + required_codes)

    @classmethod
    def _administrative_division_code_for_row(
        cls,
        row: pd.Series,
    ) -> Optional[str]:
        """Generates a dot-separated administrative code for a single DataFrame row.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame containing country, 
                admin codes, and feature details.

        Returns:
            Optional[str]: The dot-separated full administrative path string, 
                or None if the entry does not map to a valid administrative level.
        """
        level = cls._admin_level_from_feature_code(row.get("feature_code", ""))
        if level is None:
            return None
        return cls._admin_code_from_parts(
            row.get("country_code", ""),
            row.get("admin1_code", ""),
            row.get("admin2_code", ""),
            row.get("admin3_code", ""),
            row.get("admin4_code", ""),
            level=level,
        )

    def _derive_sovereign_state_country_codes(self) -> set[str]:
        """Identifies and resolves the set of active sovereign state country codes.

        Returns:
            set[str]: A set of upper-case ISO country codes determined to be 
                sovereign states, adjusted for default and custom overrides.
        """
        rows = self.df[
            (self.df["feature_class"] == "A")
            & self.df["feature_code"].isin(self.SOVEREIGN_STATE_FEATURE_CODES)
        ]
        codes = {
            str(code).upper()
            for code in rows["country_code"].tolist()
            if str(code).strip()
        }
        codes.difference_update(self.sovereign_state_overrides.keys())
        codes.update(self.sovereign_state_overrides.values())
        return codes

    def _build_administrative_division_index(self) -> dict[str, dict]:
        """Creates an optimized lookup index for administrative division records.

        Maps full dot-separated administrative codes to their respective metadata, 
        resolving duplicate record conflicts by keeping the node with the highest 
        known population.

        Returns:
            dict[str, dict]: A lookup dictionary mapping administrative division 
                codes to metadata dictionaries (containing geonameid, name, 
                coordinates, level, feature code, and population).
        """
        index: dict[str, dict] = {}
        admin_rows = self.df[
            (self.df["feature_class"] == "A")
            & self.df["feature_code"].isin(
                self.ADMINISTRATIVE_DIVISION_FEATURE_CODES
            )
        ]

        for _, row in admin_rows.iterrows():
            admin_code = self._administrative_division_code_for_row(row)
            if not admin_code:
                continue

            existing = index.get(admin_code)
            population = float(row.get("population", 0) or 0)
            if existing is not None and existing.get("population", 0) >= population:
                continue

            index[admin_code] = {
                "geonameid": int(row["geonameid"]),
                "name": str(row["name"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "admin_level": self._admin_level_from_feature_code(
                    row["feature_code"]
                ),
                "feature_code": str(row["feature_code"]),
                "population": population,
            }

        return index
    
    def _build_fast_indices(self) -> None:
        """Constructs an optimized inverted lookup index from location names to GeoNames IDs.

        Iterates through the dataset to map primary, ASCII, and alternate name 
        variants (including normalized and diacritic-folded representations) to 
        their corresponding identifiers, then sets the DataFrame index to geonameid.
        """
        name_to_ids: dict[str, set[int]] = defaultdict(set)

        for row in self.df.itertuples(index=False):
            gid = int(row.geonameid)  # type: ignore
            names = {
                self.normalize_name(row.name),  # type: ignore
                self.normalize_name(row.asciiname),  # type: ignore
            }

            if pd.notna(row.alternatenames) and row.alternatenames:  # type: ignore
                for alt in str(row.alternatenames).split(","):  # type: ignore
                    names.add(self.normalize_name(alt))

            for name in names:
                if not name:
                    continue
                name_to_ids[name].add(gid)
                folded = self.ascii_fold(name)
                if folded and folded != name:
                    name_to_ids[folded].add(gid)

        self.name_to_ids: dict[str, list[int]] = {
            name: sorted(ids)
            for name, ids in name_to_ids.items()
        }
        self.df.set_index("geonameid", inplace=True, drop=False)

    def _is_settlement(self, row: pd.Series) -> bool:
        """Determines if a DataFrame record represents a settlement.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            bool: True if the entry belongs to the populated place feature class.
        """
        return row["feature_class"] == self.SETTLEMENT_FEATURE_CLASS

    def _is_administrative_division(self, row: pd.Series) -> bool:
        """Determines if a DataFrame record represents an administrative division.

        Matches traditional subnational units, region-like zones, non-sovereign 
        political entities, and sovereign states that fall under explicit overrides.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            bool: True if the record meets operational administrative division criteria.
        """
        feature_class = row["feature_class"]
        feature_code = row["feature_code"]
        country_code = str(row.get("country_code", "")).upper()

        if (
            feature_class == "L"
            and feature_code in self.REGIONLIKE_FEATURE_CODES
        ):
            return True

        if feature_class != "A":
            return False
        if feature_code in self.ADMINISTRATIVE_DIVISION_FEATURE_CODES:
            return True
        if feature_code in self.NON_SOVEREIGN_POLITICAL_FEATURE_CODES:
            return True
        return (
            feature_code in self.SOVEREIGN_STATE_FEATURE_CODES
            and country_code in self.sovereign_state_overrides
        )

    def _is_sovereign_state(self, row: pd.Series) -> bool:
        """Determines if a DataFrame record represents a valid independent country.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            bool: True if the record maps to an independent political entity 
                and is verified by the sovereign state rules and filters.
        """
        country_code = str(row.get("country_code", "")).upper()
        return (
            row["feature_class"] == "A"
            and row["feature_code"] in self.SOVEREIGN_STATE_FEATURE_CODES
            and country_code in self.sovereign_state_country_codes
            and country_code not in self.sovereign_state_overrides
        )

    def _is_continent(self, row: pd.Series) -> bool:
        """Determines if a DataFrame record represents a continent.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            bool: True if the record matches the continent feature category.
        """
        return (
            row["feature_class"] == self.CONTINENT_FEATURE[0]
            and row["feature_code"] == self.CONTINENT_FEATURE[1]
        )

    def _is_valid_location_type(self, row: pd.Series) -> bool:
        """Validates if a record maps to any of the accepted operational hierarchy nodes.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            bool: True if the location qualifies as a settlement, administrative 
                division, sovereign state, or continent.
        """
        return (
            self._is_settlement(row)
            or self._is_administrative_division(row)
            or self._is_sovereign_state(row)
            or self._is_continent(row)
        )

    def _country_info(self, country_code: str) -> dict:
        """Safely extracts dictionary metadata for a given country code.

        Args:
            country_code (str): The ISO country code.

        Returns:
            dict: The matching country info attributes, or an empty dictionary if 
                missing.
        """
        return self.country_map.get(country_code, {})

    def _is_european(self, row: pd.Series) -> bool:
        """Checks whether the given location entry belongs to a European country.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            bool: True if the country maps to the European continent code.
        """
        info = self._country_info(row.get("country_code", ""))
        return info.get("continent_code") == self.EUROPE_CODE

    def _type_rank(self, row: pd.Series) -> int:
        """Assigns an operational type weight used to rank resolving location variants.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            int: A ranking integer where higher values represent broader geo-spatial 
                types (Continent=90, State=80, Admin=70, European Settlement=60, 
                Other Settlement=50, Unknown=0).
        """
        if self._is_continent(row):
            return 90
        if self._is_sovereign_state(row):
            return 80
        if self._is_administrative_division(row):
            return 70
        if self._is_settlement(row):
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
        """Calculates a comprehensive match score for a candidate location record.

        Combines name matching accuracy, structural types, local country biases, 
        base population sizes, and global pipeline context mentions.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame representing the candidate.
            query (str): The raw user or mention query string.
            support_count (int): Global cluster frequency context used as weight support.

        Returns:
            float: The aggregate score representing candidate match confidence, 
                or -1.0 if the type is completely invalid.
        """
        if not self._is_valid_location_type(row):
            return -1.0

        query_norm = self.normalize_name(query)
        exact_primary = query_norm == row["name_norm"]
        exact_ascii = query_norm == row["ascii_norm"]
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
        """Ensures a Continent instance is built and cached in the location index.

        Args:
            continent_name (str): Canonical name of the target continent.

        Returns:
            Optional[str]: The constructed unique continent ID, or None if the input 
                name is missing or invalid.
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

    def _sovereign_country_code_for(self, country_code: str) -> str:
        """Determines the correct upper-level sovereign country code, resolving overrides.

        Args:
            country_code (str): The raw country ISO code.

        Returns:
            str: The final mapped upper-case sovereign country code.
        """
        code = str(country_code or "").upper()
        return self.sovereign_state_overrides.get(code, code)

    def _get_or_create_sovereign_state(
        self,
        c_code: str,
        country_name: str,
        continent_name: str,
    ) -> Optional[str]:
        """Ensures a SovereignState instance is built and cached in the index.

        Resolves structural overrides and ensures child connectivity to the correct 
        continent wrapper.

        Args:
            c_code (str): Raw target country ISO code.
            country_name (str): Default fallback country name.
            continent_name (str): Default fallback continent name.

        Returns:
            Optional[str]: The unique sovereign state node ID, or None if the target 
                fails validation against sovereign state constraints.
        """
        if not c_code or c_code == "unknown":
            return None

        sovereign_code = self._sovereign_country_code_for(c_code)
        if sovereign_code not in self.sovereign_state_country_codes:
            return None

        sovereign_info = self._country_info(sovereign_code)
        sovereign_name = sovereign_info.get("country_name", country_name)
        sovereign_continent = sovereign_info.get("continent_name", continent_name)
        parent_id = self._get_or_create_continent(sovereign_continent)
        sovereign_state_id = f"sovereign_state_{sovereign_code}"

        if sovereign_state_id not in self.resolved_locations:
            coords = self.sovereign_state_coords.get(
                sovereign_code,
                {"latitude": 0.0, "longitude": 0.0},
            )
            self.resolved_locations[sovereign_state_id] = SovereignState(
                id=sovereign_state_id,
                name=sovereign_name,
                latitude=float(coords["latitude"]),
                longitude=float(coords["longitude"]),
                part_of=parent_id,
            )

        return sovereign_state_id

    def _get_parent_for_administrative_division(
        self,
        admin_code_full: str,
        c_code: str,
        country_name: str,
        continent_name: str,
    ) -> Optional[str]:
        """Resolves the immediate hierarchical structural parent of an administrative unit.

        Recursively resolves to a higher administrative tier if a nested path exists, 
        otherwise drops back to the standard sovereign state level.

        Args:
            admin_code_full (str): Dot-separated administrative path string.
            c_code (str): Iso code for the base country.
            country_name (str): Name of the parent country.
            continent_name (str): Name of the parent continent.

        Returns:
            Optional[str]: The parent location node string ID, or None if unresolvable.
        """
        parts = [part for part in str(admin_code_full).split(".") if part]
        if len(parts) > 2:
            parent_code = ".".join(parts[:-1])
            if (
                parent_code in self.administrative_division_index
                or parent_code in self.region_map
            ):
                return self._get_or_create_administrative_division(
                    parent_code,
                    c_code,
                    country_name,
                    continent_name,
                )

        return self._get_or_create_sovereign_state(
            c_code,
            country_name,
            continent_name,
        )

    def _get_or_create_administrative_division(
        self,
        admin_code_full: Optional[str],
        c_code: str,
        country_name: str,
        continent_name: str,
        display_name: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        admin_level: Optional[int] = None,
    ) -> Optional[str]:
        """Ensures an AdministrativeDivision entity is built and cached in the index.

        Validates the paths, tracks the depth level, hooks up recursive parenting structures, 
        and applies custom geographical coordinate and label overrides.

        Args:
            admin_code_full (Optional[str]): Dot-separated administrative path string.
            c_code (str): Base country ISO code.
            country_name (str): Fallback country name.
            continent_name (str): Fallback continent name.
            display_name (Optional[str], optional): Explicit overriding name text.
            latitude (Optional[float], optional): Explicit overriding latitude coordinate.
            longitude (Optional[float], optional): Explicit overriding longitude coordinate.
            admin_level (Optional[int], optional): Explicit overriding division depth level.

        Returns:
            Optional[str]: The unique administrative division entry string ID, 
                or the parent sovereign state ID if path codes are missing or shallow.
        """
        if (
            not admin_code_full
            or admin_code_full == "unknown"
            or str(admin_code_full).endswith(".")
        ):
            return self._get_or_create_sovereign_state(
                c_code,
                country_name,
                continent_name,
            )

        admin_code_full = str(admin_code_full)
        parts = [part for part in admin_code_full.split(".") if part]
        if len(parts) < 2:
            return self._get_or_create_sovereign_state(
                c_code,
                country_name,
                continent_name,
            )

        c_code = c_code or parts[0]
        parent_id = self._get_parent_for_administrative_division(
            admin_code_full,
            c_code,
            country_name,
            continent_name,
        )
        administrative_division_id = f"administrative_division_{admin_code_full}"

        if administrative_division_id not in self.resolved_locations:
            metadata = self.administrative_division_index.get(admin_code_full, {})
            name = (
                display_name
                or metadata.get("name")
                or self.region_map.get(admin_code_full)
                or admin_code_full
            )
            fallback_coords = self.admin1_coords.get(
                admin_code_full,
                {"latitude": 0.0, "longitude": 0.0},
            )
            lat = (
                float(latitude)
                if latitude is not None
                else float(metadata.get("latitude", fallback_coords["latitude"]))
            )
            lon = (
                float(longitude)
                if longitude is not None
                else float(metadata.get("longitude", fallback_coords["longitude"]))
            )
            level = (
                admin_level
                if admin_level is not None
                else metadata.get("admin_level")
                or max(len(parts) - 1, 1)
            )

            self.resolved_locations[administrative_division_id] = (
                AdministrativeDivision(
                    id=administrative_division_id,
                    name=str(name),
                    latitude=lat,
                    longitude=lon,
                    part_of=parent_id,
                    admin_level=int(level) if level is not None else None,
                )
            )

        return administrative_division_id

    def _deepest_administrative_division_code_for_row(
        self,
        row: pd.Series,
    ) -> Optional[str]:
        """Finds the most granular valid administrative division path code for a row.

        Checks descending tiers from level 4 down to level 1 using active administrative 
        division index data and falling back to base region maps.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            Optional[str]: The dot-separated full administrative path string if 
                resolved, otherwise None.
        """
        for level in range(4, 0, -1):
            admin_code = self._admin_code_from_parts(
                row.get("country_code", ""),
                row.get("admin1_code", ""),
                row.get("admin2_code", ""),
                row.get("admin3_code", ""),
                row.get("admin4_code", ""),
                level=level,
            )
            if not admin_code:
                continue
            if admin_code in self.administrative_division_index:
                return admin_code
            if level == 1 and admin_code in self.region_map:
                return admin_code

        return None

    def _candidate_ids_for_query(self, query: str) -> list[int]:
        """Retrieves all GeoNames IDs matching a normalized/folded query variant.

        Args:
            query (str): The raw text mention string to query.

        Returns:
            list[int]: A list of un-ordered candidate GeoNames integer identifiers.
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
        """Finds the single best evaluated candidate for a specific mention query.

        Args:
            query (str): The location mention string.
            support_count (int): Frequency count for this mention context.

        Returns:
            Optional[GeoCandidate]: The highest-scoring GeoNames candidate object 
                or None if no valid matches are found.
        """
        candidate_ids = self._candidate_ids_for_query(query)
        if not candidate_ids:
            return None

        best: Optional[GeoCandidate] = None
        for gid in candidate_ids:
            row = self.df.loc[gid]
            if not self._is_valid_location_type(row):  # type: ignore
                continue

            score = self._candidate_score(
                row,  # type: ignore
                query,
                support_count,
            )
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
        """Resolve cluster queries independently and group candidates by GeoNames ID.

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
                    matched_query=(
                        existing.matched_query
                        if existing.support_count >= best.support_count
                        else best.matched_query
                    ),
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
        """Determines if a split candidate meets the thresholds to separate into its own node.

        Args:
            candidate (GeoCandidate): The particular candidate to check.
            all_candidates (list[GeoCandidate]): The full list of competitors 
                extracted from this cluster.

        Returns:
            bool: True if the candidate has enough standalone support or high 
                enough relative score to be kept as an independent target entity.
        """
        if not all_candidates:
            return False
        if len(all_candidates) == 1:
            return True
        if candidate.support_count >= 2:
            return True

        best_score = max(c.score for c in all_candidates)
        return candidate.score >= best_score * 0.80

    def _location_id_for_row(self, row: pd.Series) -> Optional[str]:
        """Constructs a deterministic string node ID for a location entry.

        Args:
            row (pd.Series): A row from the GeoNames DataFrame.

        Returns:
            Optional[str]: A unique string identifier prefixed by location type 
                (e.g., 'settlement_12345'), or None if unresolvable.
        """
        gid = int(row["geonameid"])
        f_class = row["feature_class"]
        f_code = row["feature_code"]
        c_code = str(row.get("country_code", "")).upper()

        if self._is_settlement(row):
            return f"settlement_{gid}"

        if self._is_sovereign_state(row):
            sovereign_code = self._sovereign_country_code_for(c_code)
            return f"sovereign_state_{sovereign_code}"

        if self._is_administrative_division(row):
            admin_code = self._administrative_division_code_for_row(row)
            if admin_code:
                return f"administrative_division_{admin_code}"

            is_country_like = (
                f_class == "A"
                and (
                    f_code in self.NON_SOVEREIGN_POLITICAL_FEATURE_CODES
                    or f_code in self.SOVEREIGN_STATE_FEATURE_CODES
                )
            )
            if is_country_like and c_code:
                return f"administrative_division_{c_code}"

            return f"administrative_division_{gid}"

        if f_class == self.CONTINENT_FEATURE[0] and f_code == self.CONTINENT_FEATURE[1]:
            cont_name = str(row["name"]).replace(" ", "_").lower()
            return f"continent_{cont_name}"

        return None

    def _candidate_to_location(self, candidate: GeoCandidate) -> Optional[Location]:
        """Converts a parsed GeoCandidate back to a full object hierarchy node wrapper.

        Creates or maps cached parent entries to fully construct structural trees 
        (e.g., Settlement -> Admin Division -> Sovereign State -> Continent).

        Args:
            candidate (GeoCandidate): The matched geo candidate instance.

        Returns:
            Optional[Location]: A concrete subclass instance of Location (Settlement, 
                AdministrativeDivision, SovereignState, or Continent) or None 
                if unresolvable.
        """
        row = self.df.loc[candidate.geonameid]
        if not self._is_valid_location_type(row):  # type: ignore
            return None

        loc_id = self._location_id_for_row(row)  # type: ignore
        if loc_id is None:
            return None
        if loc_id in self.resolved_locations:
            return self.resolved_locations[loc_id]

        lat = float(row["latitude"]) # type: ignore
        lon = float(row["longitude"]) # type: ignore
        c_code = str(row.get("country_code", "")).upper()
        country_info = self.country_map.get(c_code, {})
        country_name = country_info.get("country_name", "unknown")
        continent_name = country_info.get("continent_name", "unknown")
        display_name = str(row["name"])

        if self._is_settlement(row):  # type: ignore
            admin_code = self._deepest_administrative_division_code_for_row(
                row,  # type: ignore
            )
            parent_id = self._get_or_create_administrative_division(
                admin_code,
                c_code,
                country_name,
                continent_name,
            )
            loc = Settlement(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
                part_of=parent_id,
            )

        elif self._is_sovereign_state(row):  # type: ignore
            parent_id = self._get_or_create_continent(continent_name)
            loc = SovereignState(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
                part_of=parent_id,
            )

        elif self._is_administrative_division(row):  # type: ignore
            admin_code = self._administrative_division_code_for_row(
                row,  # type: ignore
            )
            if admin_code:
                admin_id = self._get_or_create_administrative_division(
                    admin_code,
                    c_code,
                    country_name,
                    continent_name,
                    display_name=display_name,
                    latitude=lat,
                    longitude=lon,
                    admin_level=self._admin_level_from_feature_code(
                        row.get("feature_code", ""),
                    ),
                )
                if admin_id is None:
                    return None
                return self.resolved_locations.get(admin_id)

            parent_id: Optional[str]
            if (
                row["feature_class"] == "L"
                and row["feature_code"] in self.REGIONLIKE_FEATURE_CODES
            ): # type: ignore
                parent_admin_code = self._deepest_administrative_division_code_for_row(
                    row,  # type: ignore
                )
                parent_id = self._get_or_create_administrative_division(
                    parent_admin_code,
                    c_code,
                    country_name,
                    continent_name,
                )
            else:
                parent_id = self._get_or_create_sovereign_state(
                    c_code,
                    country_name,
                    continent_name,
                )

            loc = AdministrativeDivision(
                id=loc_id,
                name=display_name,
                latitude=lat,
                longitude=lon,
                part_of=parent_id,
                admin_level=None,
            )

        elif self._is_continent(row):  # type: ignore
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
        """Resolves raw location mentions belonging to a cluster into physical locations.

        Evaluates potential candidates, performs a soft-clustering split decision, 
        and extracts distinct locations with their structural parent hierarchies.

        Args:
            queries (list[str]): Raw string location mentions to process.
            mention_counts (Optional[Mapping[str, int]]): Global context count metrics 
                mapping mention tokens to their global occurrences.

        Returns:
            list[Location]: A deduplicated list of structured Location subclass 
                instances successfully resolved from the string cluster.
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
            if loc is None or loc.id in seen_ids:
                continue
            seen_ids.add(loc.id)
            locations.append(loc)

        return locations

    def get_all_locations(self) -> list[Location]:
        """Retrieves all unified spatial locations cached across runtime cycles.

        Returns:
            list[Location]: The list of currently resolved and instantiated 
                Location objects.
        """
        return list(self.resolved_locations.values())