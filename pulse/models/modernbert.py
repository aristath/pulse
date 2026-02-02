import logging
import threading
import torch
from transformers import AutoTokenizer

from pulse.models.base import (
    BaseModel, get_torch_device, is_latin_text,
    HAS_OPENVINO, load_sequence_classification_model,
)

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = 0.3


class ModernBERTNLI(BaseModel):
    """ModernBERT-large-zeroshot-v2.0: NLI-based zero-shot classification."""

    name = "modernbert-nli"

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        model_id = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
        self._device = get_torch_device()
        backend = "OpenVINO GPU" if HAS_OPENVINO else str(self._device)
        logger.info("Loading %s on %s...", model_id, backend)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = load_sequence_classification_model(model_id, device="GPU")
        logger.info("%s loaded on %s", self.name, backend)

    DEFAULT_COUNTRY = "This article is about {country}."
    DEFAULT_SECTOR = "This is relevant to the {sector} sector."
    DEFAULT_SENTIMENT = "This is good news for the {sector} sector in {country}."

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

        text = self.truncate(text, 6000)
        country_tpl = prompt_country or self.DEFAULT_COUNTRY
        sector_tpl = prompt_sector or self.DEFAULT_SECTOR
        sentiment_tpl = prompt_sentiment or self.DEFAULT_SENTIMENT

        # Pass 1: country relevance
        country_hypotheses = [country_tpl.format(country=c) for c in countries]
        scores = self._nli_batch(text, country_hypotheses)
        relevant = [c for c, s in zip(countries, scores) if s >= RELEVANCE_THRESHOLD]

        if not relevant:
            return {}

        # Pass 2: sector relevance (once, shared across countries)
        all_sectors = set()
        for country in relevant:
            all_sectors.update(sectors.get(country, sectors.get("global", [])))
        all_sectors = sorted(all_sectors)

        sector_hypotheses = [sector_tpl.format(sector=s) for s in all_sectors]
        sector_scores = self._nli_batch(text, sector_hypotheses)
        relevant_sectors = {s for s, sc in zip(all_sectors, sector_scores) if sc >= RELEVANCE_THRESHOLD}

        if not relevant_sectors:
            return {}

        # Pass 3: sentiment only for relevant sectors
        signals = {}
        for country in relevant:
            country_sectors = [s for s in sectors.get(country, sectors.get("global", [])) if s in relevant_sectors]
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

    def _infer(self, inputs):
        """Run inference with a lock to prevent concurrent OpenVINO access."""
        with self._lock:
            if not HAS_OPENVINO:
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    return self._model(**inputs).logits
            return self._model(**inputs).logits

    def _nli_batch(self, premise: str, hypotheses: list[str]) -> list[float]:
        """NLI — return entailment scores, one hypothesis at a time."""
        scores = []
        for hyp in hypotheses:
            inputs = self._tokenizer(
                premise, hyp, return_tensors="pt", truncation=True, max_length=4096,
            )
            probs = torch.softmax(self._infer(inputs), dim=-1)
            scores.append(probs[0, 0].item())
        return scores

    def _nli_batch_full(
        self, premise: str, hypotheses: list[str]
    ) -> tuple[list[float], list[float]]:
        """NLI — return (entailment_scores, contradiction_scores), one hypothesis at a time."""
        entail, contra = [], []
        for hyp in hypotheses:
            inputs = self._tokenizer(
                premise, hyp, return_tensors="pt", truncation=True, max_length=4096,
            )
            probs = torch.softmax(self._infer(inputs), dim=-1)
            entail.append(probs[0, 0].item())
            contra.append(probs[0, 1].item())
        return entail, contra
