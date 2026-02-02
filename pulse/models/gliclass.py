import logging
import threading

import torch
from transformers import AutoTokenizer
from gliclass import GLiClassModel

from pulse.models.base import BaseModel, is_latin_text

logger = logging.getLogger(__name__)

MODEL_ID = "knowledgator/GLiClass-modern-base-v3.0"
RELEVANCE_THRESHOLD = 0.3
SENTIMENT_THRESHOLD = 0.2


class GLiClassNLI(BaseModel):
    """GLiClass-modern-base-v3.0: zero-shot classification via label tags.

    All labels processed in a single forward pass instead of per-hypothesis NLI.
    Uses torch.compile with OpenVINO backend for GPU acceleration when available,
    falls back to plain PyTorch CPU.
    """

    name = "gliclass"

    def __init__(self, name: str = "gliclass"):
        self.name = name
        self._tokenizer = None
        self._model = None
        self._prompt_first = True
        self._lock = threading.Lock()

    def load(self):
        logger.info("Loading %s...", MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = GLiClassModel.from_pretrained(MODEL_ID)
        self._model.eval()
        self._prompt_first = getattr(self._model.config, "prompt_first", True)
        logger.info("GLiClass loaded on CPU")

    def _infer(self, text: str, labels: list[str]) -> list[float]:
        """Prepend <<LABEL>> tags, tokenize, run model, sigmoid -> scores."""
        tag_str = "".join(f"<<LABEL>>{lbl}" for lbl in labels) + "<<SEP>>"
        full = tag_str + text if self._prompt_first else text + tag_str

        inputs = self._tokenizer(
            full, return_tensors="pt", truncation=True, max_length=4096,
        )

        with self._lock:
            with torch.no_grad():
                out = self._model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                logits = out.logits

        return torch.sigmoid(logits[0, : len(labels)]).tolist()

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

        # Pass 1: country relevance
        country_labels = [f"This article is about {c}" for c in countries]
        scores = self._infer(text, country_labels)
        relevant = [c for c, s in zip(countries, scores) if s >= RELEVANCE_THRESHOLD]
        if not relevant:
            return {}

        # Pass 2: sector relevance (once, shared across countries)
        all_sectors = set()
        for country in relevant:
            all_sectors.update(sectors.get(country, sectors.get("global", [])))
        all_sectors = sorted(all_sectors)

        sector_tpl = prompt_sector or "This is relevant to the {sector} sector"
        sector_labels = [sector_tpl.format(sector=s) for s in all_sectors]
        sector_scores = self._infer(text, sector_labels)
        relevant_sectors = {s for s, sc in zip(all_sectors, sector_scores) if sc >= RELEVANCE_THRESHOLD}

        if not relevant_sectors:
            return {}

        # Pass 3: sentiment only for relevant sectors
        signals: dict = {}
        for country in relevant:
            country_sectors = [s for s in sectors.get(country, sectors.get("global", [])) if s in relevant_sectors]
            if not country_sectors:
                continue

            labels = []
            for sector in country_sectors:
                labels.append(f"positive for {country} {sector}")
                labels.append(f"negative for {country} {sector}")

            label_scores = self._infer(text, labels)

            country_signals: dict = {}
            for i, sector in enumerate(country_sectors):
                pos = label_scores[i * 2]
                neg = label_scores[i * 2 + 1]
                if pos >= SENTIMENT_THRESHOLD or neg >= SENTIMENT_THRESHOLD:
                    sentiment = max(-1.0, min(1.0, round(pos - neg, 4)))
                    country_signals[sector] = sentiment

            if country_signals:
                signals[country.lower()] = country_signals

        return signals

    def validate_company(self, text: str, company_name: str) -> float:
        """Score how relevant the article is to the given company."""
        text = self.truncate(text, 6000)
        scores = self._infer(text, [f"This article is about {company_name}"])
        return scores[0]

    def score_company_sentiment(
        self, text: str, company_name: str, prompt: str = "",
    ) -> tuple[float, float]:
        """Return (sentiment, impact) for the company.

        sentiment = pos - neg, clamped to [-1, 1].
        impact    = max(pos, neg).
        """
        text = self.truncate(text, 6000)
        scores = self._infer(
            text,
            [f"positive for {company_name}", f"negative for {company_name}"],
        )
        pos, neg = scores[0], scores[1]
        sentiment = max(-1.0, min(1.0, round(pos - neg, 4)))
        impact = round(max(pos, neg), 4)
        return sentiment, impact
