import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pulse.models.base import BaseModel, get_torch_device, is_latin_text

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = 0.3
CHUNK_SIZE = 400  # words per chunk (DeBERTa max 512 tokens)


class DeBERTaNLI(BaseModel):
    """DeBERTa-v3-large-mnli: Highest accuracy NLI, uses chunking for long texts."""

    name = "deberta-nli"

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def load(self):
        model_id = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
        self._device = get_torch_device()
        logger.info("Loading %s on %s...", model_id, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()
        logger.info("%s loaded on %s", self.name, self._device)

    DEFAULT_COUNTRY = "This article is about {country}."
    DEFAULT_SENTIMENT = "This is good news for the {sector} sector in {country}."

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

        chunks = self._chunk_text(text)
        country_tpl = prompt_country or self.DEFAULT_COUNTRY
        sentiment_tpl = prompt_sentiment or self.DEFAULT_SENTIMENT

        # Pass 1: batch country relevance per chunk, take max across chunks
        country_hypotheses = [country_tpl.format(country=c) for c in countries]
        best_scores = [0.0] * len(countries)
        for chunk in chunks:
            scores = self._nli_batch(chunk, country_hypotheses)
            best_scores = [max(b, s) for b, s in zip(best_scores, scores)]

        relevant = [
            c for c, s in zip(countries, best_scores) if s >= RELEVANCE_THRESHOLD
        ]
        if not relevant:
            return {}

        # Use only the first chunk for pass 2 (speed vs accuracy tradeoff)
        best_chunk = chunks[0]

        # Pass 2: batch sector sentiment for each relevant country
        signals = {}
        for country in relevant:
            country_sectors = sectors.get(country, sectors.get("global", []))
            if not country_sectors:
                continue

            hypotheses = [
                sentiment_tpl.format(sector=sector, country=country)
                for sector in country_sectors
            ]
            entail_scores, contra_scores = self._nli_batch_full(best_chunk, hypotheses)

            country_signals = {}
            for sector, ent, con in zip(country_sectors, entail_scores, contra_scores):
                if ent >= SENTIMENT_THRESHOLD or con >= SENTIMENT_THRESHOLD:
                    sentiment = round(ent - con, 4)
                    sentiment = max(-1.0, min(1.0, sentiment))
                    country_signals[sector] = sentiment

            if country_signals:
                signals[country.lower()] = country_signals

        return signals

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that fit DeBERTa's 512-token context."""
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [text]
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE):
            chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        return chunks

    def _nli_batch(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Batch NLI — return entailment scores."""
        inputs = self._tokenizer(
            [premise] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # v2.0: 2-class (0=entailment, 1=not_entailment)
        return probs[:, 0].tolist()

    def _nli_batch_full(
        self, premise: str, hypotheses: list[str]
    ) -> tuple[list[float], list[float]]:
        """Batch NLI — return (entailment_scores, contradiction_scores)."""
        inputs = self._tokenizer(
            [premise] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # v2.0: 2-class (0=entailment, 1=not_entailment)
        return probs[:, 0].tolist(), probs[:, 1].tolist()
