import logging
import torch

logger = logging.getLogger(__name__)

DEFAULT_HYPOTHESIS = "This text is about business or economics."


class ImpactScorer:
    """Single-hypothesis NLI scorer using shared ModernBERT weights.

    Returns a float 0.0–1.0 representing the entailment probability
    that the article has significant financial market impact.
    """

    name = "impact"

    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def ready(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def set_model(self, tokenizer, model):
        """Reuse already-loaded ModernBERT tokenizer and model."""
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Impact scorer: sharing ModernBERT weights")

    def score(self, text: str, hypothesis: str = "") -> float:
        """Return impact score 0.0 to 1.0 for the given article text."""
        hypothesis = hypothesis or DEFAULT_HYPOTHESIS

        words = text.split()
        if len(words) > 6000:
            text = " ".join(words[:6000])

        inputs = self._tokenizer(
            text,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # ModernBERT zeroshot v2.0: 2-class (0=entailment, 1=not_entailment)
        return round(probs[0, 0].item(), 4)
