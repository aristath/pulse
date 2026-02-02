import logging
import re
import threading
import torch
from transformers import AutoTokenizer

from pulse.models.base import (
    BaseModel, get_torch_device, is_latin_text,
    HAS_OPENVINO, load_sequence_classification_model,
)

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
        self._lock = threading.Lock()

    def load(self):
        model_id = "ProsusAI/finbert"
        self._device = get_torch_device()
        backend = "OpenVINO GPU" if HAS_OPENVINO else str(self._device)
        logger.info("Loading %s on %s...", model_id, backend)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = load_sequence_classification_model(model_id, device="GPU")
        logger.info("%s loaded on %s", self.name, backend)

    def classify(
        self,
        text: str,
        countries: list[str],
        sectors: dict[str, list[str]],
        prompt_country: str = "",
        prompt_sentiment: str = "",
        prompt_sector: str = "",
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
                if re.search(r'\b' + re.escape(sector.lower()) + r'\b', text_lower):
                    country_signals[sector] = round(avg_sentiment, 4)

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
        )
        with self._lock:
            if not HAS_OPENVINO:
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    logits = self._model(**inputs).logits
            else:
                logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # FinBERT: [positive, negative, neutral]
        positive = probs[0, 0].item()
        negative = probs[0, 1].item()
        return positive - negative
