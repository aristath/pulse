from abc import ABC, abstractmethod
import logging

import torch

try:
    from optimum.intel import OVModelForSequenceClassification
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

logger = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Return the best available torch device: MPS (Apple), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_latin_text(text: str, threshold: float = 0.5) -> bool:
    """Check if text is predominantly Latin script.

    Returns False for Arabic, CJK, Cyrillic, etc. which cause
    extremely slow tokenization on English-trained models.
    """
    if not text:
        return False
    sample = text[:500]
    latin = sum(1 for c in sample if c.isascii() or '\u00C0' <= c <= '\u024F')
    return latin / len(sample) >= threshold


def load_sequence_classification_model(model_id: str, device: str = "CPU", openvino_cache: str | None = None):
    """Load model via OpenVINO on the given device if available, else PyTorch.

    Args:
        model_id: HuggingFace model ID.
        device: OpenVINO device target ("GPU", "CPU"). Ignored when falling back to PyTorch.
        openvino_cache: Optional path for cached IR files.
    """
    if HAS_OPENVINO:
        from pathlib import Path
        cache_dir = str(Path(openvino_cache or f"./ov_models/{model_id.replace('/', '_')}").resolve())
        try:
            model = OVModelForSequenceClassification.from_pretrained(cache_dir, compile=False)
            model.to(device.lower())
            model.compile()
            logger.info("Loaded OpenVINO model on %s from cache: %s", device, cache_dir)
            return model
        except Exception:
            pass
        try:
            logger.info("Converting %s to OpenVINO IR...", model_id)
            model = OVModelForSequenceClassification.from_pretrained(model_id, export=True, compile=False)
            model.save_pretrained(cache_dir)
            model.to(device.lower())
            model.compile()
            logger.info("Saved OpenVINO IR to %s, compiled on %s", cache_dir, device)
            return model
        except Exception:
            pass
        # GPU may reject dynamic shapes; retry with static batch=1
        if device.upper() == "GPU":
            try:
                model = OVModelForSequenceClassification.from_pretrained(cache_dir, compile=False)
                model.reshape(1, 512)
                model.to(device.lower())
                model.compile()
                logger.info("Loaded OpenVINO model on %s (static shapes) from cache: %s", device, cache_dir)
                return model
            except Exception as exc:
                logger.warning("OpenVINO failed for %s on %s: %s, falling back to PyTorch", model_id, device, exc)
        else:
            logger.warning("OpenVINO failed for %s on %s, falling back to PyTorch", model_id, device)

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(get_torch_device())
    model.eval()
    return model


class BaseModel(ABC):
    """Base interface for all classification models."""

    name: str  # unique model identifier, e.g. 'deberta'

    @abstractmethod
    def load(self):
        """Load model weights into memory."""
        ...

    @abstractmethod
    def classify(
        self,
        text: str,
        countries: list[str],
        sectors: dict[str, list[str]],
        prompt_country: str = "",
        prompt_sentiment: str = "",
        prompt_sector: str = "",
    ) -> dict[str, dict[str, float]]:
        """
        Three-pass classification:
        1. Determine relevant countries from `countries` list.
        2. Determine relevant sectors.
        3. For each relevant country, classify sentiment per relevant sector.

        Args:
            text: Article content.
            countries: List of country names to check relevance.
            sectors: Dict mapping country to list of sector names.
            prompt_country: Template for country relevance, with {country} placeholder.
            prompt_sentiment: Template for sentiment, with {sector} and {country} placeholders.
            prompt_sector: Template for sector relevance, with {sector} placeholder.

        Returns:
            Nested dict: {country: {sector: sentiment_score}}.
            Sentiment score is -1.0 to 1.0.
        """
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Rough truncation by whitespace tokens."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])
