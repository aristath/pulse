import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pulse.models.base import BaseModel, get_torch_device, is_latin_text

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.3
CHUNK_SIZE = 400  # words per chunk (FinBERT max 512 tokens)


class FinBERT(BaseModel):
    """ProsusAI/finbert: Financial sentiment analysis.

    FinBERT only does sentiment (positive/negative/neutral), so we apply it
    differently: run sentiment per chunk, then attribute sentiment to countries
    and sectors found via keyword matching in the text.
    """

    name = "finbert"

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def load(self):
        model_id = "ProsusAI/finbert"
        self._device = get_torch_device()
        logger.info("Loading %s on %s...", model_id, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()
        logger.info("%s loaded on %s", self.name, self._device)

    def classify(
        self,
        text: str,
        countries: list[str],
        sectors: dict[str, list[str]],
        prompt_country: str = "",
        prompt_sentiment: str = "",
    ) -> dict:
        if not is_latin_text(text):
            return {}

        text_lower = text.lower()

        # Pass 1: Which countries are mentioned?
        relevant = [c for c in countries if c.lower() in text_lower]
        if not relevant:
            return {}

        # Pass 2: Overall financial sentiment from FinBERT
        chunks = self._chunk_text(text)
        sentiments = [self._get_sentiment(chunk) for chunk in chunks]
        # Average sentiment across chunks
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        if abs(avg_sentiment) < 0.05:
            return {}

        # Attribute sentiment to relevant countries and their sectors (keyword match)
        signals = {}
        for country in relevant:
            country_sectors = sectors.get(country, sectors.get("global", []))
            if not country_sectors:
                continue

            country_signals = {}
            for sector in country_sectors:
                if sector.lower() in text_lower:
                    country_signals[sector] = round(avg_sentiment, 4)

            # If no specific sectors matched, assign to first sector as "general"
            if not country_signals and country_sectors:
                country_signals[country_sectors[0]] = round(avg_sentiment, 4)

            if country_signals:
                signals[country.lower()] = country_signals

        return signals

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [text]
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE):
            chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        return chunks

    def _get_sentiment(self, text: str) -> float:
        """Return sentiment score: positive (0 to 1), negative (-1 to 0)."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # FinBERT: [positive, negative, neutral]
        positive = probs[0, 0].item()
        negative = probs[0, 1].item()
        return positive - negative
