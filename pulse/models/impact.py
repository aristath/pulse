import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pulse.models.base import get_torch_device, is_latin_text

logger = logging.getLogger(__name__)

DEFAULT_HYPOTHESIS = "This text is about business or economics."

MODEL_ID = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"


class ImpactScorer:
    """Dedicated ModernBERT NLI scorer for impact scoring.

    Loads its own copy of ModernBERT so it can run in parallel with
    the classify workers without lock contention.

    Returns a float 0.0–1.0 representing the entailment probability
    that the article has significant financial market impact.
    """

    name = "impact"

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None

    @property
    def ready(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def load(self):
        """Load a dedicated ModernBERT instance for impact scoring."""
        self._device = get_torch_device()
        logger.info("Impact scorer: loading %s on %s...", MODEL_ID, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self._model.to(self._device)
        self._model.eval()
        logger.info("Impact scorer: loaded on %s", self._device)

    def set_model(self, tokenizer, model):
        """Reuse already-loaded ModernBERT tokenizer and model (fallback)."""
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Impact scorer: sharing ModernBERT weights")

    def score(self, text: str, hypothesis: str = "") -> float:
        """Return impact score 0.0 to 1.0 for the given article text."""
        if not is_latin_text(text):
            return 0.0

        hypothesis = hypothesis or DEFAULT_HYPOTHESIS

        words = text.split()
        if len(words) > 6000:
            text = " ".join(words[:6000])

        device = self._device or get_torch_device()
        inputs = self._tokenizer(
            text,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # ModernBERT zeroshot v2.0: 2-class (0=entailment, 1=not_entailment)
        return round(probs[0, 0].item(), 4)
