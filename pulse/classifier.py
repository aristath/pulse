import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import psutil

from pulse.models.base import BaseModel
from pulse.models.modernbert import ModernBERTNLI
from pulse.models.deberta import DeBERTaNLI
from pulse.models.finbert import FinBERT
from pulse.models.impact import ImpactScorer
from pulse.database import (
    get_unprocessed_article_for_model, get_next_article_for_impact,
    get_setting, save_result, save_impact, update_article_date,
)
from pulse.fetcher import fetch_article_content

logger = logging.getLogger(__name__)

# Per-model processing status for web UI: {model_name: {article_id, article_url, started_at}}
processing_status = {}

# Current thermal throttle delay (seconds) — exposed for the UI
thermal_throttle: dict = {"delay": 0.0, "temp": None}


def _thermal_delay() -> float:
    """Compute inference pause based on CPU package temperature.

    0s below 75°C, 5s at 75°C, +5s per degree above that.
    """
    temps = getattr(psutil, "sensors_temperatures", lambda: None)()
    if not temps:
        return 0.0
    temp = None
    for chip in ("coretemp", "k10temp"):
        if chip in temps:
            for entry in temps[chip]:
                if "package" in (entry.label or "").lower():
                    temp = entry.current
                    break
            if temp is None and temps[chip]:
                temp = temps[chip][0].current
            if temp is not None:
                break
    if temp is None and "x86_pkg_temp" in temps and temps["x86_pkg_temp"]:
        temp = temps["x86_pkg_temp"][0].current
    thermal_throttle["temp"] = temp
    if temp is None or temp < 75:
        thermal_throttle["delay"] = 0.0
        return 0.0
    delay = (temp - 74) * 5.0
    thermal_throttle["delay"] = delay
    return delay


class EnsembleClassifier:
    def __init__(self):
        self._models: list[BaseModel] = [
            ModernBERTNLI(),
            DeBERTaNLI(),
            FinBERT(),
        ]
        self._impact_scorer = ImpactScorer()
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._loaded = False
        self._countries: list[str] = []
        self._sectors: dict[str, list[str]] = {}

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self._models]

    def load_models(self):
        """Load all models into memory. Call once at startup."""
        logger.info("Loading %d models...", len(self._models))
        failed = []
        for model in self._models:
            try:
                model.load()
            except Exception:
                logger.exception("Failed to load model %s", model.name)
                failed.append(model)
        for m in failed:
            self._models.remove(m)
        if self._models:
            self._loaded = True
            for model in self._models:
                processing_status[model.name] = {
                    "article_id": None,
                    "article_url": None,
                    "started_at": None,
                }
            # Share ModernBERT weights with impact scorer
            modernbert = self._get_model("modernbert-nli")
            if modernbert:
                self._impact_scorer.set_model(modernbert._tokenizer, modernbert._model)
                processing_status[self._impact_scorer.name] = {
                    "article_id": None,
                    "article_url": None,
                    "started_at": None,
                }
            logger.info("Loaded %d models (%d failed)", len(self._models), len(failed))
        else:
            logger.error("No models loaded successfully")

    async def fetch_labels(self) -> bool:
        """Fetch geographies and industries from Sentinel API.

        Returns True if labels were fetched, False otherwise.
        """
        sentinel_url = await get_setting("sentinel_url")
        if not sentinel_url:
            logger.warning("Sentinel URL not configured")
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{sentinel_url}/api/pulse/labels")
                resp.raise_for_status()
                data = resp.json()

            self._countries = data.get("geographies", [])
            industries = data.get("industries", [])
            self._sectors = {c: industries for c in self._countries}

            logger.info(
                "Labels from Sentinel: %d geographies, %d industries",
                len(self._countries), len(industries),
            )
            return True
        except Exception:
            logger.exception("Failed to fetch labels from Sentinel")
            return False

    async def classify_next_for_model(self, model_name: str) -> bool:
        """Pick an unprocessed article for one model, classify it, save result.

        Returns True if an article was processed, False if none available.
        """
        if not self._loaded:
            return False

        model = self._get_model(model_name)
        if not model:
            return False

        if not self._countries:
            await self.fetch_labels()
            if not self._countries:
                logger.warning("No labels available — configure Sentinel URL")
                return False

        article = await get_unprocessed_article_for_model(model_name)
        if not article:
            return False

        # Fetch configurable prompts
        prompt_country = await get_setting("prompt_country") or ""
        prompt_sentiment = await get_setting("prompt_sentiment") or ""

        processing_status[model_name] = {
            "article_id": article["id"],
            "article_url": article["url"],
            "started_at": time.time(),
        }

        logger.info("[%s] Classifying article %d: %s", model_name, article["id"], article["url"])

        content, published_at = await fetch_article_content(article["url"])
        if published_at and not article.get("published_at"):
            await update_article_date(article["id"], published_at)

        if not content:
            logger.warning("[%s] Could not fetch content for article %d", model_name, article["id"])
            await save_result(article["id"], model_name, {})
            self._clear_model_status(model_name)
            return True

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                model.classify,
                content,
                self._countries,
                self._sectors,
                prompt_country,
                prompt_sentiment,
            )
            await save_result(article["id"], model_name, result)
            if result:
                logger.info("[%s] Article %d: %s", model_name, article["id"], result)
        except Exception as e:
            logger.error("[%s] Failed on article %d: %s", model_name, article["id"], e)
            await save_result(article["id"], model_name, {})

        self._clear_model_status(model_name)
        delay = _thermal_delay()
        if delay > 0:
            logger.info("[%s] Thermal throttle: %.0fs pause (CPU %.0f°C)", model_name, delay, thermal_throttle["temp"])
            await asyncio.sleep(delay)
        return True

    async def score_next_impact(self) -> bool:
        """Pick an unscored article, compute its impact, save it.

        Returns True if an article was scored, False if none available.
        """
        if not self._impact_scorer.ready:
            return False

        article = await get_next_article_for_impact()
        if not article:
            return False

        prompt_impact = await get_setting("prompt_impact") or ""

        processing_status[self._impact_scorer.name] = {
            "article_id": article["id"],
            "article_url": article["url"],
            "started_at": time.time(),
        }

        logger.info("[impact] Scoring article %d: %s", article["id"], article["url"])

        content, published_at = await fetch_article_content(article["url"])
        if published_at and not article.get("published_at"):
            await update_article_date(article["id"], published_at)

        if not content:
            logger.warning("[impact] Could not fetch content for article %d", article["id"])
            await save_impact(article["id"], 0.0)
            self._clear_model_status(self._impact_scorer.name)
            return True

        loop = asyncio.get_event_loop()
        try:
            score = await loop.run_in_executor(
                self._executor,
                self._impact_scorer.score,
                content,
                prompt_impact,
            )
            await save_impact(article["id"], score)
            logger.info("[impact] Article %d: %.4f", article["id"], score)
        except Exception as e:
            logger.error("[impact] Failed on article %d: %s", article["id"], e)
            await save_impact(article["id"], 0.0)

        self._clear_model_status(self._impact_scorer.name)
        delay = _thermal_delay()
        if delay > 0:
            logger.info("[impact] Thermal throttle: %.0fs pause (CPU %.0f°C)", delay, thermal_throttle["temp"])
            await asyncio.sleep(delay)
        return True

    def _get_model(self, name: str) -> BaseModel | None:
        for m in self._models:
            if m.name == name:
                return m
        return None

    def _clear_model_status(self, model_name: str):
        processing_status[model_name] = {
            "article_id": None,
            "article_url": None,
            "started_at": None,
        }


# Singleton
ensemble = EnsembleClassifier()
