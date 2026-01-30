import logging
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

from pulse.models.base import BaseModel

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.3
SENTIMENT_THRESHOLD = 0.2


class GLiClassLarge(BaseModel):
    name = "gliclass-large"

    def __init__(self):
        self._pipeline = None

    def load(self):
        model_id = "knowledgator/GLiClass-modern-large-v2.0"
        logger.info("Loading %s...", model_id)
        model = GLiClassModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            model=model, tokenizer=tokenizer, classification_type="multi-label", device="cpu"
        )
        logger.info("%s loaded", self.name)

    def classify(self, text: str, countries: list[str], sectors: dict[str, list[str]]) -> dict:
        text = self.truncate(text, 6000)
        return _gliclass_classify(self._pipeline, text, countries, sectors)


class GLiClassBase(BaseModel):
    name = "gliclass-base"

    def __init__(self):
        self._pipeline = None

    def load(self):
        model_id = "knowledgator/GLiClass-modern-base-v3.0"
        logger.info("Loading %s...", model_id)
        model = GLiClassModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            model=model, tokenizer=tokenizer, classification_type="multi-label", device="cpu"
        )
        logger.info("%s loaded", self.name)

    def classify(self, text: str, countries: list[str], sectors: dict[str, list[str]]) -> dict:
        text = self.truncate(text, 6000)
        return _gliclass_classify(self._pipeline, text, countries, sectors)


def _gliclass_classify(pipeline, text, countries, sectors):
    """Shared two-pass classification logic for GLiClass models."""
    # Pass 1: Which countries are relevant?
    country_labels = [f"This article is about {c}" for c in countries]
    results = pipeline(text, country_labels, multi_label=True)
    relevant = []
    for label, score in zip(results["labels"], results["scores"]):
        if score >= RELEVANCE_THRESHOLD:
            # Extract country name from label
            country = label.replace("This article is about ", "")
            relevant.append(country)

    if not relevant:
        return {}

    # Pass 2: For each relevant country, classify sectors + sentiment
    signals = {}
    for country in relevant:
        country_sectors = sectors.get(country, sectors.get("global", []))
        if not country_sectors:
            continue

        labels = []
        for sector in country_sectors:
            labels.append(f"positive for {country} {sector}")
            labels.append(f"negative for {country} {sector}")

        results = pipeline(text, labels, multi_label=True)
        country_signals = {}
        for label, score in zip(results["labels"], results["scores"]):
            if score < SENTIMENT_THRESHOLD:
                continue
            if label.startswith("positive for"):
                sector = label.replace(f"positive for {country} ", "")
                country_signals[sector] = country_signals.get(sector, 0) + score
            elif label.startswith("negative for"):
                sector = label.replace(f"negative for {country} ", "")
                country_signals[sector] = country_signals.get(sector, 0) - score

        # Clamp to [-1, 1]
        for s in country_signals:
            country_signals[s] = max(-1.0, min(1.0, round(country_signals[s], 4)))

        if country_signals:
            signals[country.lower()] = country_signals

    return signals
