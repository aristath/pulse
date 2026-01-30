import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pulse.models.base import BaseModel

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = 0.3


class ModernBERTNLI(BaseModel):
    """ModernBERT-large-zeroshot-v2.0: NLI-based zero-shot classification."""

    name = "modernbert-nli"

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def load(self):
        model_id = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
        logger.info("Loading %s...", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.eval()
        logger.info("%s loaded", self.name)

    DEFAULT_COUNTRY = "This article is about {country}."
    DEFAULT_SENTIMENT = "This is good news for the {sector} sector in {country}."

    def classify(self, text: str, countries: list[str], sectors: dict[str, list[str]],
                 prompt_country: str = "", prompt_sentiment: str = "") -> dict:
        text = self.truncate(text, 6000)
        country_tpl = prompt_country or self.DEFAULT_COUNTRY
        sentiment_tpl = prompt_sentiment or self.DEFAULT_SENTIMENT

        # Pass 1: batch all country hypotheses in one forward pass
        country_hypotheses = [country_tpl.format(country=c) for c in countries]
        scores = self._nli_batch(text, country_hypotheses)
        relevant = [c for c, s in zip(countries, scores) if s >= RELEVANCE_THRESHOLD]

        if not relevant:
            return {}

        # Pass 2: batch all sector hypotheses for relevant countries
        signals = {}
        for country in relevant:
            country_sectors = sectors.get(country, sectors.get("global", []))
            if not country_sectors:
                continue

            hypotheses = [
                sentiment_tpl.format(sector=sector, country=country)
                for sector in country_sectors
            ]
            entail_scores, contra_scores = self._nli_batch_full(text, hypotheses)

            country_signals = {}
            for sector, ent, con in zip(country_sectors, entail_scores, contra_scores):
                if ent >= SENTIMENT_THRESHOLD or con >= SENTIMENT_THRESHOLD:
                    sentiment = round(ent - con, 4)
                    sentiment = max(-1.0, min(1.0, sentiment))
                    country_signals[sector] = sentiment

            if country_signals:
                signals[country.lower()] = country_signals

        return signals

    def _nli_batch(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Batch NLI — return entailment scores for all hypotheses."""
        inputs = self._tokenizer(
            [premise] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=8192,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # v2.0: 2-class (0=entailment, 1=not_entailment)
        return probs[:, 0].tolist()

    def _nli_batch_full(self, premise: str, hypotheses: list[str]) -> tuple[list[float], list[float]]:
        """Batch NLI — return (entailment_scores, contradiction_scores)."""
        inputs = self._tokenizer(
            [premise] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=8192,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # v2.0: 2-class (0=entailment, 1=not_entailment)
        return probs[:, 0].tolist(), probs[:, 1].tolist()
