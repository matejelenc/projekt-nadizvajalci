"""Code for clustering locations to clusters likely representing the same location."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from typing import Mapping
import math
import re
import unicodedata

DIRECTIONAL_MODIFIERS = {
    "severna", "severni", "severno", "severne", "sever",
    "južna", "južni", "južno", "južne", "jug",
    "vzhodna", "vzhodni", "vzhodno", "vzhodne", "vzhod",
    "zahodna", "zahodni", "zahodno", "zahodne", "zahod",
    "srednja", "srednji", "srednje", 
    "centralna", "centralni", "centralno", "centralne",
}

DISTINGUISHING_MODIFIERS = {
    "nova", "novo", "novi", "nove",
    "stara", "staro", "stari", "stare",
    "ekvatorialna", "ekvatorialni", "ekvatorialno",
    "hercegovina",
    "demokratična",
    "ljudska",
    "zvezna",
    "federativna",
    "federacija",
    "republika",
    "kraljevina",
}

WEAK_ADMIN_WORDS = {
    "republika",
    "država",
    "dežela",
    "pokrajina",
    "regija",
    "mesto",
    "občina",
    "jezero",
    "reka",
}

RISKY_MODIFIERS = DIRECTIONAL_MODIFIERS | DISTINGUISHING_MODIFIERS

@dataclass(frozen=True)
class ClusterConfig:
    """
    Configuration parameters for the location clustering algorithm.

    Attributes:
        similarity_threshold (float): Threshold for overall similarity.
        prefix_len (int): Length of prefix for blocking.
        char_ngram_size (int): Size of character n-grams.
        max_bucket_size (int): Maximum size of a bucket during blocking.
        min_prefix_chars (int): Minimum characters required for prefix scoring.
        complete_link_threshold (float): Threshold for complete linkage clustering.
        risky_containment_threshold (float): Threshold above which risky containment is evaluated.
        min_count (int): Minimum frequency of a mention to be considered.
        rare_count (int): Frequency threshold for rare mentions.
        rare_mention_min_length (int): Minimum length for rare mentions.
        frequency_boost_strength (float): Strength of the frequency boost in similarity.
        lowercase (bool): Whether to convert text to lowercase.
        strip_diacritics (bool): Whether to remove diacritics.
        remove_numeric_tokens (bool): Whether to remove tokens containing only numbers.
        allow_risky_containment (bool): Whether to allow merging when risky containment modifiers are present.
    """
    similarity_threshold: float = 0.86

    # Blocking parameters.
    prefix_len: int = 5
    char_ngram_size: int = 4
    max_bucket_size: int = 1000

    # Scoring parameters.
    min_prefix_chars: int = 4
    complete_link_threshold: float = 0.82
    risky_containment_threshold: float = 0.97

    # Frequency handling.
    min_count: int = 1
    rare_count: int = 1
    rare_mention_min_length: int = 3
    frequency_boost_strength: float = 0.03

    # Normalization.
    lowercase: bool = True
    strip_diacritics: bool = False
    remove_numeric_tokens: bool = True

    # Safety.
    allow_risky_containment: bool = False


@dataclass(frozen=True)
class SimilarityResult:
    """
    Data class containing the result of a similarity comparison.

    Attributes:
        score (float): The calculated similarity score.
        reason (str): The primary reason or method that produced the score.
        risky (bool): Flag indicating if the similarity match is considered risky.
    """
    score: float
    reason: str
    risky: bool = False


@dataclass
class MentionCluster:
    """
    Represents a clustered group of location mentions.

    Attributes:
        cluster_id (int): Unique identifier for the cluster.
        mentions (set[str]): Set of raw mention strings belonging to this cluster.
        canonical_name (str): The representative name for the cluster.
        total_count (int): Total frequency count of all mentions in the cluster.
        confidence (float): Confidence score of the cluster formation.
        merge_reasons (Counter[str]): Counter of reasons why mentions were merged.
    """
    cluster_id: int
    mentions: set[str]
    canonical_name: str
    total_count: int
    confidence: float
    merge_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def size(self) -> int:
        """
        Get the number of unique mentions in the cluster.

        Returns:
            int: The number of mentions.
        """
        return len(self.mentions)


class UnionFind:
    """
    Disjoint-set data structure for efficient union-find operations.
    """
    def __init__(self, n: int):
        """
        Initialize the UnionFind structure.

        Args:
            n (int): Number of elements in the set.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        Find the root representative of the set containing element x with path compression.

        Args:
            x (int): The element to find.

        Returns:
            int: The root representative of the element.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """
        Unite the sets containing elements a and b using union by rank.

        Args:
            a (int): The first element.
            b (int): The second element.
        """
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def normalize_text(text: str, config: ClusterConfig) -> str:
    """
    Normalize text according to the provided configuration.

    Args:
        text (str): The text string to normalize.
        config (ClusterConfig): Configuration parameters for normalization.

    Returns:
        str: The normalized text string.
    """
    text = text.strip()

    if config.lowercase:
        text = text.lower()

    if config.strip_diacritics:
        text = "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize(text: str, config: ClusterConfig) -> list[str]:
    """
    Tokenize text into a list of words, optionally removing numeric tokens.

    Args:
        text (str): The text to tokenize.
        config (ClusterConfig): Configuration determining tokenization rules.

    Returns:
        list[str]: A list of tokens extracted from the text.
    """
    tokens = [tok for tok in text.split() if tok]

    if config.remove_numeric_tokens:
        non_numeric = [tok for tok in tokens if not tok.isdigit()]
        if non_numeric:
            tokens = non_numeric

    return tokens


def common_prefix_len(a: str, b: str) -> int:
    """
    Calculate the length of the common prefix between two strings.

    Args:
        a (str): The first string.
        b (str): The second string.

    Returns:
        int: The number of matching characters at the start of both strings.
    """
    n = min(len(a), len(b))
    i = 0

    while i < n and a[i] == b[i]:
        i += 1

    return i


def sequence_ratio(a: str, b: str) -> float:
    """
    Calculate the sequence similarity ratio between two strings.

    Args:
        a (str): The first string.
        b (str): The second string.

    Returns:
        float: A similarity ratio between 0.0 and 1.0.
    """
    return SequenceMatcher(None, a, b).ratio()


def char_ngrams(text: str, n: int) -> set[str]:
    """
    Generate character n-grams from a given text.

    Args:
        text (str): The input string.
        n (int): The length of the n-grams to generate.

    Returns:
        set[str]: A set of generated character n-grams.
    """
    compact = text.replace(" ", "")

    if not compact:
        return set()

    if len(compact) <= n:
        return {compact}

    return {
        compact[i:i + n]
        for i in range(len(compact) - n + 1)
    }


def jaccard(a: set[str], b: set[str]) -> float:
    """
    Calculate the Jaccard similarity between two sets.

    Args:
        a (set[str]): The first set.
        b (set[str]): The second set.

    Returns:
        float: The Jaccard similarity score between 0.0 and 1.0.
    """
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def single_token_similarity(a: str, b: str, config: ClusterConfig) -> SimilarityResult:
    """
    Compute the similarity between two individual tokens.

    Args:
        a (str): The first token.
        b (str): The second token.
        config (ClusterConfig): Configuration parameters for scoring.

    Returns:
        SimilarityResult: The resulting similarity score and reason for the score.
    """
    if a == b:
        return SimilarityResult(1.0, "exact_token")

    seq = sequence_ratio(a, b)

    prefix_len = common_prefix_len(a, b)
    if prefix_len >= config.min_prefix_chars:
        prefix_score = prefix_len / min(len(a), len(b))
    else:
        prefix_score = 0.0

    ng_score = jaccard(
        char_ngrams(a, config.char_ngram_size),
        char_ngrams(b, config.char_ngram_size),
    )

    score = max(seq, prefix_score, ng_score)

    if score == prefix_score and prefix_score > 0:
        reason = "prefix"
    elif score == ng_score:
        reason = "char_ngram"
    else:
        reason = "sequence"

    return SimilarityResult(score, reason)


def soft_token_alignment_score(
    a_tokens: list[str],
    b_tokens: list[str],
    config: ClusterConfig,
) -> float:
    """
    Calculate a soft alignment similarity score between two lists of tokens.

    Args:
        a_tokens (list[str]): The first list of tokens.
        b_tokens (list[str]): The second list of tokens.
        config (ClusterConfig): Configuration parameters for scoring.

    Returns:
        float: The computed soft alignment score.
    """
    if not a_tokens or not b_tokens:
        return 0.0

    used_b: set[int] = set()
    matched_scores: list[float] = []

    for tok_a in a_tokens:
        best_score = 0.0
        best_j = None

        for j, tok_b in enumerate(b_tokens):
            if j in used_b:
                continue

            score = single_token_similarity(tok_a, tok_b, config).score

            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            used_b.add(best_j)
            matched_scores.append(best_score)

    if not matched_scores:
        return 0.0

    return sum(matched_scores) / max(len(a_tokens), len(b_tokens))


def best_token_match_score(
    short_tokens: list[str],
    long_tokens: list[str],
    config: ClusterConfig,
) -> float:
    """
    Compute the best average match score of a shorter list of tokens against a longer list.

    Args:
        short_tokens (list[str]): The list containing fewer tokens.
        long_tokens (list[str]): The list containing more tokens.
        config (ClusterConfig): Configuration parameters for scoring.

    Returns:
        float: The average of the best match scores for each token in the short list.
    """
    if not short_tokens or not long_tokens:
        return 0.0

    scores = []

    for short_tok in short_tokens:
        best = max(
            single_token_similarity(short_tok, long_tok, config).score
            for long_tok in long_tokens
        )
        scores.append(best)

    return sum(scores) / len(scores)


def has_risky_containment(short_tokens: list[str], long_tokens: list[str]) -> bool:
    """
    Check if the longer token list contains risky modifiers not present in the short list.

    Args:
        short_tokens (list[str]): The shorter token list.
        long_tokens (list[str]): The longer token list.

    Returns:
        bool: True if risky modifiers are present in the extra tokens, False otherwise.
    """
    short_set = set(short_tokens)
    long_set = set(long_tokens)

    extra_tokens = long_set - short_set

    return bool(extra_tokens & RISKY_MODIFIERS)

def mention_similarity(a: str, b: str, config: ClusterConfig) -> SimilarityResult:
    """
    Compute the overall similarity score between two location mentions.

    Args:
        a (str): The first mention string.
        b (str): The second mention string.
        config (ClusterConfig): Configuration parameters for clustering.

    Returns:
        SimilarityResult: The calculated similarity score, reason, and risk assessment.
    """
    a_norm = normalize_text(a, config)
    b_norm = normalize_text(b, config)

    if a_norm == b_norm:
        return SimilarityResult(1.0, "exact")

    a_tokens = tokenize(a_norm, config)
    b_tokens = tokenize(b_norm, config)

    if not a_tokens or not b_tokens:
        return SimilarityResult(0.0, "empty")

    # Single-word case: important for Slovene inflection-like endings.
    if len(a_tokens) == 1 and len(b_tokens) == 1:
        result = single_token_similarity(a_tokens[0], b_tokens[0], config)
        return SimilarityResult(result.score, f"single_{result.reason}")

    a_joined = " ".join(a_tokens)
    b_joined = " ".join(b_tokens)

    string_score = sequence_ratio(a_joined, b_joined)
    token_score = soft_token_alignment_score(a_tokens, b_tokens, config)

    if len(a_tokens) <= len(b_tokens):
        short_tokens, long_tokens = a_tokens, b_tokens
    else:
        short_tokens, long_tokens = b_tokens, a_tokens

    containment_score = best_token_match_score(short_tokens, long_tokens, config)
    risky = has_risky_containment(short_tokens, long_tokens)

    if risky and not config.allow_risky_containment:
        containment_score = min(containment_score, 0.80)

    scores = {
        "string": string_score,
        "token_alignment": token_score,
        "containment": containment_score,
    }

    reason, score = max(scores.items(), key=lambda x: x[1])

    return SimilarityResult(score, reason, risky=risky)


def frequency_adjusted_score(
    base_score: float,
    count_a: int,
    count_b: int,
    config: ClusterConfig,
) -> float:
    """
    Apply a small score boost when both mentions are frequent.

    This boost is intentionally small so that frequency does not override 
    poor string similarity evidence.

    Args:
        base_score (float): The initial similarity score.
        count_a (int): Frequency count of the first mention.
        count_b (int): Frequency count of the second mention.
        config (ClusterConfig): Configuration parameters for clustering.

    Returns:
        float: The frequency-adjusted similarity score bounded between 0.0 and 1.0.
    """
    if count_a <= 0 or count_b <= 0:
        return base_score

    weaker = min(count_a, count_b)
    stronger = max(count_a, count_b)

    # In [0, 1], higher when both mentions are frequent and not extremely imbalanced.
    frequency_signal = math.log1p(weaker) / math.log1p(stronger)

    boosted = base_score + config.frequency_boost_strength * frequency_signal

    return min(1.0, boosted)


def blocking_keys(mention: str, config: ClusterConfig) -> set[str]:
    """
    Generate blocking keys for a mention to group potential candidate pairs.

    Args:
        mention (str): The mention string to generate keys for.
        config (ClusterConfig): Configuration parameters for blocking.

    Returns:
        set[str]: A set of blocking keys associated with the mention.
    """
    norm = normalize_text(mention, config)
    tokens = tokenize(norm, config)

    keys: set[str] = set()

    if not norm or not tokens:
        return keys

    compact = norm.replace(" ", "")

    keys.add(f"name_prefix:{compact[:config.prefix_len]}")
    keys.add(f"first:{tokens[0][:config.prefix_len]}")
    keys.add(f"last:{tokens[-1][:config.prefix_len]}")

    for tok in tokens:
        if len(tok) >= config.min_prefix_chars:
            keys.add(f"tok_prefix:{tok[:config.prefix_len]}")
        else:
            keys.add(f"tok_exact:{tok}")

    for ng in char_ngrams(norm, config.char_ngram_size):
        keys.add(f"ng:{ng}")

    for tok in tokens:
        if tok not in WEAK_ADMIN_WORDS and len(tok) >= config.min_prefix_chars:
            keys.add(f"strong_token:{tok[:config.prefix_len]}")

    return keys


def should_keep_mention(
    mention: str,
    count: int,
    config: ClusterConfig,
) -> bool:
    """
    Filter obvious low-value, short, or rare noise mentions.

    This function is intentionally conservative and mainly removes very rare, 
    very short noise from the dataset.

    Args:
        mention (str): The mention string to evaluate.
        count (int): Frequency count of the mention.
        config (ClusterConfig): Configuration parameters for filtering.

    Returns:
        bool: True if the mention should be kept, False otherwise.
    """
    if count < config.min_count:
        return False

    norm = normalize_text(mention, config)
    tokens = tokenize(norm, config)

    if not tokens:
        return False

    compact = "".join(tokens)

    if count <= config.rare_count and len(compact) < config.rare_mention_min_length:
        return False

    return True


def generate_candidate_pairs(
    mentions: list[str],
    mention_counts: Mapping[str, int],
    config: ClusterConfig,
) -> set[tuple[int, int]]:
    """
    Generate a set of candidate index pairs for similarity comparison.

    Uses blocking keys to avoid O(N^2) comparisons across all mentions.

    Args:
        mentions (list[str]): List of unique mention strings.
        mention_counts (Mapping[str, int]): Dictionary mapping mentions to their frequency.
        config (ClusterConfig): Configuration parameters.

    Returns:
        set[tuple[int, int]]: A set of tuples containing indices of candidate pairs.
    """
    key_to_indices: dict[str, list[int]] = defaultdict(list)

    for i, mention in enumerate(mentions):
        for key in blocking_keys(mention, config):
            key_to_indices[key].append(i)

    candidate_pairs: set[tuple[int, int]] = set()

    for bucket in key_to_indices.values():
        if len(bucket) < 2:
            continue

        if len(bucket) > config.max_bucket_size:
            continue

        # Compare more frequent mentions first.
        bucket = sorted(
            bucket,
            key=lambda i: mention_counts[mentions[i]],
            reverse=True,
        )

        for i, j in combinations(bucket, 2):
            candidate_pairs.add((min(i, j), max(i, j)))

    return candidate_pairs


def choose_canonical_name(
    cluster: set[str],
    mention_counts: Mapping[str, int],
    config: ClusterConfig,
) -> str:
    """
    Choose a canonical representative mention for a cluster.

    The selection preference is:
    1. Higher frequency.
    2. Fewer numeric characters.
    3. More complete multiword form, unless it only adds numeric noise.
    4. Shorter normalized name as a fallback.

    Args:
        cluster (set[str]): Set of mention strings belonging to the cluster.
        mention_counts (Mapping[str, int]): Dictionary mapping mentions to their frequency.
        config (ClusterConfig): Configuration parameters.

    Returns:
        str: The selected canonical name for the cluster.
    """

    def score_name(name: str) -> tuple[int, int, int, int, str]:
        norm = normalize_text(name, config)
        tokens = tokenize(norm, config)

        count = mention_counts.get(name, 0)
        digit_count = sum(ch.isdigit() for ch in norm)
        token_count = len(tokens)

        return (
            count,
            -digit_count,
            token_count,
            -len(norm),
            name,
        )

    return max(cluster, key=score_name)


def refine_component_complete_link(
    component: set[int],
    mentions: list[str],
    mention_counts: Mapping[str, int],
    pair_scores: dict[tuple[int, int], SimilarityResult],
    config: ClusterConfig,
) -> list[set[int]]:
    """
    Refine a connected component using complete-linkage clustering to prevent transitive chain errors.

    Example of transitive chain error avoided by this function:
        A ~ B
        B ~ C
        A !~ C
    Union-find would join A, B, C. Complete-link refinement prevents this if A and C are not similar.

    Args:
        component (set[int]): Set of indices representing a roughly clustered component.
        mentions (list[str]): List of all unique mention strings.
        mention_counts (Mapping[str, int]): Dictionary mapping mentions to their frequency.
        pair_scores (dict[tuple[int, int], SimilarityResult]): Cached similarity scores between pairs.
        config (ClusterConfig): Configuration parameters for clustering.

    Returns:
        list[set[int]]: A list of refined subclusters (sets of indices).
    """

    ordered = sorted(
        component,
        key=lambda i: mention_counts[mentions[i]],
        reverse=True,
    )

    subclusters: list[set[int]] = []

    for idx in ordered:
        placed = False

        for subcluster in subclusters:
            compatible_scores = []

            for other in subcluster:
                pair = (min(idx, other), max(idx, other))

                if pair in pair_scores:
                    result = pair_scores[pair]
                    base_score = result.score
                else:
                    result = mention_similarity(
                        mentions[idx],
                        mentions[other],
                        config,
                    )
                    base_score = result.score

                score = frequency_adjusted_score(
                    base_score,
                    mention_counts[mentions[idx]],
                    mention_counts[mentions[other]],
                    config,
                )

                compatible_scores.append(score)

            if compatible_scores and min(compatible_scores) >= config.complete_link_threshold:
                subcluster.add(idx)
                placed = True
                break

        if not placed:
            subclusters.append({idx})

    return subclusters

def cluster_location_mentions(
    mention_counts: Mapping[str, int],
    *,
    similarity: float = 0.86,
    prefix_len: int = 5,
    char_ngram_size: int = 4,
    max_bucket_size: int = 1000,
    min_count: int = 1,
    rare_count: int = 1,
    rare_mention_min_length: int = 3,
    strip_diacritics: bool = False,
    allow_risky_containment: bool = False,
    frequency_boost_strength: float = 0.03,
) -> list[MentionCluster]:
    """
    Perform fully automatic pre-GeoNames clustering on location mentions.

    Args:
        mention_counts (Mapping[str, int]): Mapping from mention string to total corpus count.
            Example: {"Slovenija": 1500, "Sloveniji": 800, "Severna Makedonija": 30}
        similarity (float, optional): Base pairwise similarity threshold. 
            Recommended: 0.90 (conservative), 0.86 (balanced), 0.80 (aggressive). Defaults to 0.86.
        prefix_len (int, optional): Prefix length used for blocking keys. Defaults to 5.
        char_ngram_size (int, optional): N-gram size for character-level similarity. Defaults to 4.
        max_bucket_size (int, optional): Maximum allowed size for a blocking bucket. Defaults to 1000.
        min_count (int, optional): Mentions with a count below this are removed. Defaults to 1.
        rare_count (int, optional): Frequency threshold used with rare_mention_min_length to filter noise. Defaults to 1.
        rare_mention_min_length (int, optional): Minimum length required for mentions with rare_count. Defaults to 3.
        strip_diacritics (bool, optional): Whether to remove diacritics during normalization. Defaults to False.
        allow_risky_containment (bool, optional): If False, prevents automatic merges like "Koreja" ~ "Severna Koreja". Defaults to False.
        frequency_boost_strength (float, optional): Small score boost for frequent mentions. Recommended 0.00 to 0.05. Defaults to 0.03.

    Returns:
        list[MentionCluster]: A list of organized MentionCluster objects ready for final GeoNames resolution.
    """

    config = ClusterConfig(
        similarity_threshold=similarity,
        prefix_len=prefix_len,
        char_ngram_size=char_ngram_size,
        max_bucket_size=max_bucket_size,
        min_count=min_count,
        rare_count=rare_count,
        rare_mention_min_length=rare_mention_min_length,
        strip_diacritics=strip_diacritics,
        allow_risky_containment=allow_risky_containment,
        frequency_boost_strength=frequency_boost_strength,
    )

    cleaned_counts: dict[str, int] = {}

    for mention, count in mention_counts.items():
        if mention is None: continue
        mention = str(mention).strip()
        if not mention: continue

        count = int(count)
        if count <= 0: continue

        if should_keep_mention(mention, count, config):
            cleaned_counts[mention] = cleaned_counts.get(mention, 0) + count

    unique_mentions = sorted(
        cleaned_counts,
        key=lambda m: (-cleaned_counts[m], m),
    )

    if not unique_mentions:
        return []

    if len(unique_mentions) == 1:
        only = unique_mentions[0]
        return [
            MentionCluster(
                cluster_id=0,
                mentions={only},
                canonical_name=only,
                total_count=cleaned_counts[only],
                confidence=1.0,
                merge_reasons=Counter({"single": 1}),
            )
        ]

    candidate_pairs = generate_candidate_pairs(
        unique_mentions,
        cleaned_counts,
        config,
    )

    uf = UnionFind(len(unique_mentions))
    pair_scores: dict[tuple[int, int], SimilarityResult] = {}

    for i, j in candidate_pairs:
        a = unique_mentions[i]
        b = unique_mentions[j]

        result = mention_similarity(a, b, config)

        adjusted = frequency_adjusted_score(
            result.score,
            cleaned_counts[a],
            cleaned_counts[b],
            config,
        )

        adjusted_result = SimilarityResult(
            score=adjusted,
            reason=result.reason,
            risky=result.risky,
        )

        pair_scores[(i, j)] = adjusted_result

        threshold = config.similarity_threshold

        if result.risky:
            threshold = max(threshold, config.risky_containment_threshold)

        if adjusted >= threshold:
            uf.union(i, j)

    rough_components: dict[int, set[int]] = defaultdict(set)

    for i in range(len(unique_mentions)):
        rough_components[uf.find(i)].add(i)

    refined_components: list[set[int]] = []

    for component in rough_components.values():
        if len(component) <= 2:
            refined_components.append(component)
        else:
            refined_components.extend(
                refine_component_complete_link(
                    component,
                    unique_mentions,
                    cleaned_counts,
                    pair_scores,
                    config,
                )
            )

    clusters: list[MentionCluster] = []

    for component in refined_components:
        cluster_mentions = {unique_mentions[i] for i in component}
        canonical = choose_canonical_name(cluster_mentions, cleaned_counts, config)
        total_count = sum(cleaned_counts[m] for m in cluster_mentions)

        merge_reasons: Counter[str] = Counter()
        scores: list[float] = []

        if len(component) == 1:
            confidence = 1.0
            merge_reasons["single"] += 1
        else:
            for i, j in combinations(sorted(component), 2):
                pair = (min(i, j), max(i, j))
                result = pair_scores.get(pair)

                if result is None:
                    raw = mention_similarity(unique_mentions[i], unique_mentions[j], config)
                    score = frequency_adjusted_score(
                        raw.score,
                        cleaned_counts[unique_mentions[i]],
                        cleaned_counts[unique_mentions[j]],
                        config,
                    )
                    result = SimilarityResult(score, raw.reason, raw.risky)

                scores.append(result.score)
                merge_reasons[result.reason] += 1

            confidence = min(scores) if scores else 1.0

        clusters.append(
            MentionCluster(
                cluster_id=-1,
                mentions=cluster_mentions,
                canonical_name=canonical,
                total_count=total_count,
                confidence=confidence,
                merge_reasons=merge_reasons,
            )
        )

    clusters.sort(
        key=lambda c: (
            -c.total_count,
            -c.size,
            c.canonical_name,
        )
    )

    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i

    return clusters