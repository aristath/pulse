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
from pulse.models.gliner import CompanyScanner
from pulse.database import (
    get_unprocessed_article_for_model, get_next_article_for_impact,
    get_setting, save_result, save_impact, update_article_date,
    save_alias_scan, get_unscanned_article_for_alias,
    save_company_mention, get_next_unscored_company_result,
    save_company_sentiment,
)
from pulse.fetcher import fetch_article_content

logger = logging.getLogger(__name__)

# Per-model processing status for web UI: {model_name: {article_id, article_url, started_at}}
processing_status = {}

# Current thermal throttle delay (seconds) — exposed for the UI
thermal_throttle: dict = {"delay": 0.0, "temp": None}

# --- Worker infrastructure ---
_workers_running = False
_worker_tasks: list[asyncio.Task] = []
IDLE_SLEEP = 2.0

# Each worker pulls tasks in priority order.
# Capabilities: (task_type, model_name_or_None)
WORKER_CONFIGS = [
    ("modernbert-nli", [("impact", None), ("classify", "modernbert-nli"), ("company_sentiment", "modernbert-nli")]),
    ("deberta-nli",    [("classify", "deberta-nli"), ("company_sentiment", "deberta-nli")]),
    ("finbert",        [("classify", "finbert")]),
    ("company-scanner",[("scan", None)]),
]


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
        self._company_scanner = CompanyScanner()
        self._ticker_aliases: dict[str, list[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._loaded = False
        self._countries: list[str] = []
        self._sectors: dict[str, list[str]] = {}
        self._company_sentiment_lock = asyncio.Lock()

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
            # Load company scanner
            try:
                self._company_scanner.load()
                processing_status["company-scanner"] = {
                    "article_id": None,
                    "article_url": None,
                    "started_at": None,
                }
                processing_status["company-sentiment"] = {
                    "article_id": None,
                    "article_url": None,
                    "started_at": None,
                }
            except Exception:
                logger.exception("Failed to load CompanyScanner")
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

    async def fetch_aliases(self) -> bool:
        """Fetch company aliases from Sentinel API and update scanner embeddings.

        Returns True if aliases were fetched, False otherwise.
        """
        sentinel_url = await get_setting("sentinel_url")
        if not sentinel_url:
            logger.warning("Sentinel URL not configured — skipping alias fetch")
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{sentinel_url}/api/securities/aliases")
                resp.raise_for_status()
                data = resp.json()

            ticker_aliases: dict[str, list[str]] = {}
            for entry in data:
                symbol = entry.get("symbol")
                name = entry.get("name")
                if not symbol or not name:
                    continue
                aliases = [name]
                raw_aliases = entry.get("aliases")
                if raw_aliases:
                    aliases.extend(a.strip() for a in raw_aliases.split(",") if a.strip())
                ticker_aliases[symbol] = aliases

            self._ticker_aliases = ticker_aliases

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._company_scanner.update_aliases,
                ticker_aliases,
            )

            logger.info("Aliases from Sentinel: %d tickers, %d aliases total",
                        len(ticker_aliases), len(self._company_scanner.all_aliases))
            return True
        except Exception:
            logger.exception("Failed to fetch aliases from Sentinel")
            return False

    async def scan_next_article(self) -> bool:
        """Find an article with unscanned aliases, scan it, record results.

        Returns True if an article was scanned, False if none available.
        """
        if not self._company_scanner.ready:
            return False

        all_aliases = self._company_scanner.all_aliases
        if not all_aliases:
            await self.fetch_aliases()
            all_aliases = self._company_scanner.all_aliases
            if not all_aliases:
                return False

        # Find an article that hasn't been scanned for any alias yet.
        # We use the first alias as a proxy — all aliases are recorded together.
        article = await get_unscanned_article_for_alias(all_aliases[0])
        if not article:
            return False

        processing_status["company-scanner"] = {
            "article_id": article["id"],
            "article_url": article["url"],
            "started_at": time.time(),
        }

        logger.info("[company-scanner] Scanning article %d: %s", article["id"], article["url"])

        content, published_at = await fetch_article_content(article["url"])
        if published_at and not article.get("published_at"):
            await update_article_date(article["id"], published_at)

        matched_aliases: list[str] = []
        if content:
            loop = asyncio.get_event_loop()
            try:
                matched_aliases = await loop.run_in_executor(
                    self._executor,
                    self._company_scanner.scan,
                    content,
                )
            except Exception as e:
                logger.error("[company-scanner] Failed on article %d: %s", article["id"], e)

        # Record alias_scans for ALL aliases (marks article as fully scanned)
        for alias in all_aliases:
            await save_alias_scan(article["id"], alias)

        # Save company mentions for matched aliases
        matched_tickers = set()
        alias_to_ticker = self._company_scanner.alias_to_ticker
        for alias in matched_aliases:
            ticker = alias_to_ticker.get(alias)
            if ticker and ticker not in matched_tickers:
                matched_tickers.add(ticker)
                await save_company_mention(article["id"], ticker)

        if matched_tickers:
            logger.info("[company-scanner] Article %d: matched %s", article["id"], matched_tickers)

        self._clear_model_status("company-scanner")
        delay = _thermal_delay()
        if delay > 0:
            await asyncio.sleep(delay)
        return True

    async def classify_next_company_sentiment(self, model_name: str | None = None) -> bool:
        """Pick an unscored company_results row, run NLI sentiment, save.

        Returns True if a row was scored, False if none available.
        Guarded by a lock so only one worker processes at a time.
        """
        if not self._loaded:
            return False

        # Multiple workers may have this capability — skip if another is active
        if self._company_sentiment_lock.locked():
            return False

        async with self._company_sentiment_lock:
            row = await get_next_unscored_company_result()
            if not row:
                return False

            article_id = row["article_id"]
            ticker = row["ticker"]
            article_url = row["article_url"]

            # Find company name from ticker_aliases
            company_name = ticker
            for name_candidate in self._ticker_aliases.get(ticker, []):
                company_name = name_candidate
                break

            prompt_company = await get_setting("prompt_company") or ""

            processing_status["company-sentiment"] = {
                "article_id": article_id,
                "article_url": article_url,
                "started_at": time.time(),
            }

            logger.info("[company-sentiment] Scoring %s for article %d", ticker, article_id)

            content, _ = await fetch_article_content(article_url)
            if not content:
                logger.warning("[company-sentiment] Could not fetch content for article %d", article_id)
                await save_company_sentiment(article_id, ticker, 0.0, 0.0)
                self._clear_model_status("company-sentiment")
                return True

            # Use specified model, or fall back to any available NLI model
            model = None
            if model_name:
                model = self._get_model(model_name)
            if not model:
                model = self._get_model("modernbert-nli") or self._get_model("deberta-nli")
            if not model:
                self._clear_model_status("company-sentiment")
                return False

            tpl = prompt_company or "This is good news for {company}."
            hypothesis = tpl.format(company=company_name)

            loop = asyncio.get_event_loop()
            try:
                entail_scores, contra_scores = await loop.run_in_executor(
                    self._executor,
                    model._nli_batch_full,
                    model.truncate(content, 6000),
                    [hypothesis],
                )
                sentiment = round(entail_scores[0] - contra_scores[0], 4)
                sentiment = max(-1.0, min(1.0, sentiment))
                impact = round(max(entail_scores[0], contra_scores[0]), 4)
                await save_company_sentiment(article_id, ticker, sentiment, impact)
                logger.info("[company-sentiment] %s article %d: sentiment=%.4f impact=%.4f",
                            ticker, article_id, sentiment, impact)
            except Exception as e:
                logger.error("[company-sentiment] Failed for %s article %d: %s", ticker, article_id, e)
                await save_company_sentiment(article_id, ticker, 0.0, 0.0)

            self._clear_model_status("company-sentiment")
            delay = _thermal_delay()
            if delay > 0:
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

    async def _worker_loop(self, worker_name: str, capabilities: list[tuple[str, str | None]]):
        """Pull-based worker: try each capability in priority order, execute first available, sleep if none."""
        # Wait for models to load
        while _workers_running and not self._loaded:
            await asyncio.sleep(1.0)

        logger.info("[worker:%s] Started", worker_name)
        try:
            while _workers_running:
                did_work = False
                for task_type, model in capabilities:
                    if not _workers_running:
                        break
                    try:
                        if task_type == "classify" and model:
                            did_work = await self.classify_next_for_model(model)
                        elif task_type == "impact":
                            did_work = await self.score_next_impact()
                        elif task_type == "scan":
                            did_work = await self.scan_next_article()
                        elif task_type == "company_sentiment":
                            did_work = await self.classify_next_company_sentiment(model_name=model)
                    except Exception:
                        logger.exception("[worker:%s] Error in %s", worker_name, task_type)
                    if did_work:
                        break

                if not did_work:
                    await asyncio.sleep(IDLE_SLEEP)
        except asyncio.CancelledError:
            pass

        logger.info("[worker:%s] Stopped", worker_name)


# Singleton
ensemble = EnsembleClassifier()


def start_workers():
    """Launch all pull-based worker loops as async tasks."""
    global _workers_running
    _workers_running = True
    for worker_name, capabilities in WORKER_CONFIGS:
        task = asyncio.create_task(
            ensemble._worker_loop(worker_name, capabilities),
            name=f"worker:{worker_name}",
        )
        _worker_tasks.append(task)
    logger.info("Started %d workers", len(_worker_tasks))


async def stop_workers():
    """Signal all workers to stop and wait for them to finish."""
    global _workers_running
    _workers_running = False
    for task in _worker_tasks:
        task.cancel()
    if _worker_tasks:
        await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks.clear()
    logger.info("All workers stopped")
