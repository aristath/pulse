from abc import ABC, abstractmethod


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
