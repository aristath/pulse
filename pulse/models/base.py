from abc import ABC, abstractmethod

import torch


def get_torch_device() -> torch.device:
    """Return the best available torch device: MPS (Apple), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_latin_text(text: str, threshold: float = 0.5) -> bool:
    """Check if text is predominantly Latin script.

    Returns False for Arabic, CJK, Cyrillic, etc. which cause
    extremely slow tokenization on English-trained models.
    """
    if not text:
        return False
    sample = text[:500]
    latin = sum(1 for c in sample if c.isascii() or '\u00C0' <= c <= '\u024F')
    return latin / len(sample) >= threshold


class BaseModel(ABC):
    """Base interface for all classification models."""

    name: str  # unique model identifier, e.g. 'gliclass-large'

    @abstractmethod
    def load(self):
        """Load model weights into memory."""
        ...

    @abstractmethod
    def classify(
        self,
        text: str,
        countries: list[str],
        sectors: dict[str, list[str]],
        prompt_country: str = "",
        prompt_sentiment: str = "",
    ) -> dict[str, dict[str, float]]:
        """
        Two-pass classification:
        1. Determine relevant countries from `countries` list.
        2. For each relevant country, classify sentiment per sector.

        Args:
            text: Article content.
            countries: List of country names to check relevance.
            sectors: Dict mapping country to list of sector names.
            prompt_country: Template for country relevance, with {country} placeholder.
            prompt_sentiment: Template for sentiment, with {sector} and {country} placeholders.

        Returns:
            Nested dict: {country: {sector: sentiment_score}}.
            Sentiment score is -1.0 to 1.0.
        """
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Rough truncation by whitespace tokens."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])
