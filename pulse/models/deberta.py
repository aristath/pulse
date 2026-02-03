import logging
import math
import threading
import torch
from transformers import AutoTokenizer

from pulse.models.base import (
    BaseModel, get_torch_device, is_latin_text,
    HAS_OPENVINO, load_sequence_classification_model,
)

logger = logging.getLogger(__name__)

MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
RELEVANCE_THRESHOLD = 0.5
MAX_LENGTH = 512
CHUNK_SIZE = 400  # words per chunk (DeBERTa max 512 tokens)


class DeBERTaNLI(BaseModel):
    """DeBERTa-v3-base-zeroshot-v2.0: NLI classification on OpenVINO GPU."""

    name = "deberta-nli"

    def __init__(self, name: str = "deberta-nli"):
        self.name = name
        self._tokenizer = None
        self._model = None
        self._device = None
        self._lock = threading.Lock()

    def load(self):
        self._device = get_torch_device()
        logger.info("Loading %s ...", MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = load_sequence_classification_model(MODEL_ID, device="GPU")
        kind = "OpenVINO" if hasattr(self._model, "request") else "PyTorch"
        logger.info("%s loaded (%s)", self.name, kind)

    DEFAULT_COUNTRY = "This article is about {country}."
    DEFAULT_SECTOR = "This is relevant to the {sector} sector."
    DEFAULT_SENTIMENT_POS = "This is positive for the {sector} sector in {country}."
    DEFAULT_SENTIMENT_NEG = "This is negative for the {sector} sector in {country}."

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

        text = self.truncate(text, CHUNK_SIZE)
        country_tpl = prompt_country or self.DEFAULT_COUNTRY
        sector_tpl = prompt_sector or self.DEFAULT_SECTOR

        # Pass 1: country relevance
        country_hypotheses = [country_tpl.format(country=c) for c in countries]
        scores = self._nli_batch(text, country_hypotheses)
        relevant = [c for c, s in zip(countries, scores) if s >= RELEVANCE_THRESHOLD]
        if not relevant:
            return {}

        # Pass 2: sector relevance
        all_sectors = set()
        for country in relevant:
            all_sectors.update(sectors.get(country, sectors.get("global", [])))
        all_sectors = sorted(all_sectors)

        sector_hypotheses = [sector_tpl.format(sector=s) for s in all_sectors]
        sector_scores = self._nli_batch(text, sector_hypotheses)
        relevant_sectors = {s for s, sc in zip(all_sectors, sector_scores) if sc >= RELEVANCE_THRESHOLD}
        if not relevant_sectors:
            return {}

        # Pass 3: sentiment via separate positive/negative hypotheses
        # (2-class NLI: entailment vs not_entailment — need two hypotheses
        #  to distinguish positive from negative sentiment)
        pos_tpl = self.DEFAULT_SENTIMENT_POS
        neg_tpl = self.DEFAULT_SENTIMENT_NEG
        signals = {}
        for country in relevant:
            country_sectors = [s for s in sectors.get(country, sectors.get("global", [])) if s in relevant_sectors]
            if not country_sectors:
                continue

            pos_hyps = [pos_tpl.format(sector=s, country=country) for s in country_sectors]
            neg_hyps = [neg_tpl.format(sector=s, country=country) for s in country_sectors]
            pos_scores = self._nli_batch(text, pos_hyps)
            neg_scores = self._nli_batch(text, neg_hyps)

            country_signals = {}
            for sector, pos, neg in zip(country_sectors, pos_scores, neg_scores):
                sentiment = round(pos - neg, 4)
                sentiment = max(-1.0, min(1.0, sentiment))
                country_signals[sector] = sentiment

            if country_signals:
                signals[country.lower()] = country_signals

        return signals

    def validate_company(self, text: str, company_name: str) -> float:
        """Score how relevant the article is to the given company."""
        text = self.truncate(text, CHUNK_SIZE)
        scores = self._nli_batch(text, [f"This article is about {company_name}"])
        return scores[0]

    def score_company_sentiment(
        self, text: str, company_name: str, prompt: str = "",
    ) -> tuple[float, float]:
        """Return (sentiment, impact) for the company."""
        text = self.truncate(text, CHUNK_SIZE)
        pos = self._nli_batch(text, [f"positive for {company_name}"])[0]
        neg = self._nli_batch(text, [f"negative for {company_name}"])[0]
        sentiment = max(-1.0, min(1.0, round(pos - neg, 4)))
        impact = round(max(pos, neg), 4)
        return sentiment, impact

    def _infer(self, inputs):
        """Lock-protected forward pass."""
        with self._lock:
            if not HAS_OPENVINO:
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    return self._model(**inputs).logits
            return self._model(**inputs).logits

    def _nli_batch(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Return entailment scores, one hypothesis at a time."""
        scores = []
        for hyp in hypotheses:
            inputs = self._tokenizer(
                premise, hyp, return_tensors="pt", truncation=True,
                padding="max_length", max_length=MAX_LENGTH,
            )
            probs = torch.softmax(self._infer(inputs), dim=-1)
            score = probs[0, 0].item()
            scores.append(0.0 if math.isnan(score) else score)
        return scores

