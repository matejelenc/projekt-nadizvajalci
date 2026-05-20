"""Contains code for extracting potential location mentions using NER."""

from contextlib import contextmanager
import re
import unicodedata

import classla
import torch

def normalize_mention_key(mention: str) -> str:
    """
    Normalize a mention string to be used as a standardized key.

    Args:
        mention (str): The mention string to normalize.

    Returns:
        str: The normalized mention key.
    """
    mention = unicodedata.normalize("NFC", mention)
    mention = mention.strip()
    mention = mention.lower()
    mention = re.sub(r"\s+", " ", mention)
    return mention

@contextmanager
def torch_load_classla_compat():
    """
    Temporarily patch torch.load for CLASSLA checkpoint compatibility.

    Provides a context manager that restores the original torch.load function
    after CLASSLA models are loaded. This allows older CLASSLA checkpoints to
    load in newer PyTorch versions where weights_only may default to True.

    Yields:
        None: Control to the wrapped block while torch.load is patched.
    """
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        """
        Load a PyTorch object with weights_only disabled by default.

        Args:
            *args: Positional arguments forwarded to torch.load.
            **kwargs: Keyword arguments forwarded to torch.load.

        Returns:
            Any: Object returned by the original torch.load call.
        """
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    try:
        yield
    finally:
        torch.load = original_torch_load


class LocationExtractor:
    """
    Extract location mentions from Slovene article text using CLASSLA NER.

    Attributes:
        LABELS (list[str]): Entity labels treated as location mentions.
        IMP_LABELS (list[str]): Additional important entity labels.
        use_gpu (bool): Whether CLASSLA is configured to use GPU execution.
        nlp (classla.Pipeline): CLASSLA processing pipeline.
    """

    LABELS = ["LOC", "LOCATION"]
    IMP_LABELS = ["ORG", "GPE", "PER"]

    def __init__(self, cuda_id: int = -1):
        """
        Initialize the CLASSLA-based entity extraction pipeline.

        Args:
            cuda_id (int): ID of CUDA device on which to run CLASSLA. A negative
                value disables GPU execution.
        """
        classla.download("sl")

        self.use_gpu = cuda_id >= 0

        kwargs = {
            "processors": "tokenize,pos,lemma,ner",
            "use_gpu": self.use_gpu,
            "logging_level": "WARN",
        }

        with torch_load_classla_compat():
            self.nlp = classla.Pipeline("sl", **kwargs)

    def extract(
        self,
        contents: list[str],
        locations: dict[str, int] = dict(),
    ) -> tuple[list[list[str]], dict[str, int]]:
        """
        Extract normalized location mentions from article contents.

        Args:
            contents (list[str]): Text contents from which to extract location
                mentions.
            locations (dict[str, int]): Existing location mention counts to
                update.

        Returns:
            tuple[list[set[str]], dict[str, int]]: Per-content sets of detected
                location mentions and updated global location mention counts.
        """
        potential_mentions: list[list[str]] = []

        for content in contents:
            doc = self.nlp(content)
            content_mentions: set[str] = set()

            for ent in doc.ents:
                if ent.type in self.LABELS:

                    lemmas = [word.lemma for word in ent.words]
                    name = " ".join(lemmas)
                    name = normalize_mention_key(name)
                    content_mentions.add(name)

                    if name in locations:
                        locations[name] += 1
                    else:
                        locations[name] = 1
            
            mentions = list(content_mentions)
            potential_mentions.append(mentions)

        return potential_mentions, locations

    def __call__(
        self,
        contents: list[str],
        locations: dict[str, int] = dict(),
    ) -> tuple[list[list[str]], dict[str, int]]:
        """
        Extract normalized location mentions from article contents.

        Args:
            contents (list[str]): Text contents from which to extract location
                mentions.
            locations (dict[str, int]): Existing location mention counts to
                update.

        Returns:
            tuple[list[set[str]], dict[str, int]]: Per-content sets of detected
                location mentions and updated global location mention counts.
        """
        return self.extract(contents, locations)