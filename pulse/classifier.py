import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import httpx
import psutil

from pulse.models.base import BaseModel
from pulse.models.deberta import DeBERTaNLI
from pulse.models.impact import ImpactScorer
from pulse.models.gliner import CompanyScanner
from pulse.database import (
    get_unprocessed_article_for_model,
    get_next_article_for_impact,
    get_setting,
    set_setting,
    save_result,
    save_impact,
    update_article_date,
    get_article,
    get_company_scan_candidates,
    save_scanned_aliases,
    save_company_mention,
    get_next_unscored_company_result,
    save_company_sentiment,
    get_next_unvalidated_company_result,
    mark_company_result_validated,
    delete_company_result,
    get_articles_needing_content,
    store_article_content,
    delete_article,
)
from pulse.fetcher import fetch_article_content

logger = logging.getLogger(__name__)

# Per-model processing status for web UI: {model_name: {article_id, article_url, started_at}}
processing_status = {}

# Last 10 inference durations per worker (seconds)
worker_durations: dict[str, deque] = {}

# Thermal gate: pause all inference when CPU package temp >= 90°C,
# resume when it drops below 75°C.  Exposed for the UI.
thermal_throttle: dict = {"paused": False, "temp": None}
THERMAL_PAUSE_TEMP = 90.0
THERMAL_RESUME_TEMP = 75.0

# --- Worker infrastructure ---
_workers_running = False
_worker_tasks: list[asyncio.Task] = []
IDLE_SLEEP_NONE = 2.0  # seconds to sleep when no work available
COMPANY_SCAN_STATE_KEY = "company_scan_state"
COMPANY_SCAN_BATCH_SIZE = 200
COMPANY_SCAN_MAX_BATCHES = 6


# Each worker pulls tasks in priority order.
# Capabilities: (task_type, model_name_or_None)
WORKER_CONFIGS = [
    ("impact-1", [("impact", "impact-1")]),
    ("impact-2", [("impact", "impact-2")]),
    ("impact-3", [("impact", "impact-3")]),
    ("deberta", [("classify", "deberta")]),
    (
        "deberta-aux",
        [
            ("validate", "deberta-aux"),
            ("company_sentiment", "deberta-aux"),
        ],
    ),
    ("company-scanner", [("scan", None)]),
]

# Display labels for each processing_status key
WORKER_LABELS = {
    "impact": "Impact (ModernBERT)",
    "impact:impact-1": "Impact 1 (ModernBERT)",
    "impact:impact-2": "Impact 2 (ModernBERT)",
    "impact:impact-3": "Impact 3 (ModernBERT)",
    "classify:deberta": "Classify (DeBERTa)",
    "validate:deberta-aux": "Validate (DeBERTa)",
    "company_sentiment:deberta-aux": "Sentiment (DeBERTa)",
    "company-scanner": "Company Scanner (spaCy)",
}


def _read_cpu_temp() -> float | None:
    """Read CPU package temperature, or None if unavailable."""
    temps = getattr(psutil, "sensors_temperatures", lambda: None)()
    if not temps:
        return None
    for chip in ("coretemp", "k10temp"):
        if chip in temps:
            for entry in temps[chip]:
                if "package" in (entry.label or "").lower():
                    return entry.current
            if temps[chip]:
                return temps[chip][0].current
    if "x86_pkg_temp" in temps and temps["x86_pkg_temp"]:
        return temps["x86_pkg_temp"][0].current
    return None


async def _thermal_gate():
    """Block until CPU temperature is safe for inference.

    Pauses at >= 90°C, resumes at < 75°C.  No-op when temp is unavailable.
    """
    temp = _read_cpu_temp()
    thermal_throttle["temp"] = temp
    if temp is None:
        return

    if not thermal_throttle["paused"] and temp >= THERMAL_PAUSE_TEMP:
        thermal_throttle["paused"] = True
        logger.warning("Thermal gate: PAUSED at %.0f°C — waiting for < %.0f°C", temp, THERMAL_RESUME_TEMP)

    while thermal_throttle["paused"] and _workers_running:
        await asyncio.sleep(5)
        temp = _read_cpu_temp()
        thermal_throttle["temp"] = temp
        if temp is None or temp < THERMAL_RESUME_TEMP:
            thermal_throttle["paused"] = False
            logger.info("Thermal gate: RESUMED at %.0f°C", temp or 0)


class EnsembleClassifier:
    def __init__(self):
        self._models: list[BaseModel] = [
            DeBERTaNLI(name="deberta"),
            DeBERTaNLI(name="deberta-aux"),
        ]
        self._impact_scorers: dict[str, ImpactScorer] = {
            f"impact-{i}": ImpactScorer(name=f"impact-{i}") for i in range(1, 4)
        }
        self._company_scanner = CompanyScanner()
        self._ticker_aliases: dict[str, list[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=8)
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
            for name, scorer in list(self._impact_scorers.items()):
                try:
                    scorer.load()
                except Exception:
                    logger.exception("Failed to load %s", name)
                    del self._impact_scorers[name]
            # Load company scanner
            try:
                self._company_scanner.load()
            except Exception:
                logger.exception("Failed to load CompanyScanner")
            # Initialize processing_status from WORKER_CONFIGS
            _idle = {"article_id": None, "article_url": None, "started_at": None}
            for worker_name, capabilities in WORKER_CONFIGS:
                for task_type, model in capabilities:
                    key = f"{task_type}:{model}" if model else worker_name
                    processing_status[key] = dict(_idle)
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
                len(self._countries),
                len(industries),
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
        prompt_sector = await get_setting("prompt_sector") or ""

        status_key = f"classify:{model_name}"
        processing_status[status_key] = {
            "article_id": article["id"],
            "article_url": article["url"],
            "started_at": time.time(),
        }

        logger.info(
            "[%s] Classifying article %d: %s", model_name, article["id"], article["url"]
        )

        content = article["content"]
        if not content:
            await save_result(article["id"], model_name, {})
            self._clear_model_status(status_key)
            return True

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    model.classify,
                    content,
                    self._countries,
                    self._sectors,
                    prompt_country,
                    prompt_sentiment,
                    prompt_sector,
                ),
                timeout=300,
            )
            await save_result(article["id"], model_name, result)
            if result:
                logger.info("[%s] Article %d: %s", model_name, article["id"], result)
        except asyncio.TimeoutError:
            logger.warning("[%s] Timeout on article %d — skipping", model_name, article["id"])
            await save_result(article["id"], model_name, {})
        except Exception as e:
            logger.error("[%s] Failed on article %d: %s", model_name, article["id"], e)
            await save_result(article["id"], model_name, {})

        self._clear_model_status(status_key)
        return True

    async def score_next_impact(self, scorer_name: str | None = None) -> bool:
        """Pick an unscored article, compute its impact, save it.

        Returns True if an article was scored, False if none available.
        """
        scorer = self._impact_scorers.get(scorer_name) if scorer_name else None
        if not scorer:
            scorer = next((s for s in self._impact_scorers.values() if s.ready), None)
        if not scorer or not scorer.ready:
            return False

        article = await get_next_article_for_impact()
        if not article:
            return False
        prompt_impact = await get_setting("prompt_impact") or ""

        status_key = f"impact:{scorer.name}"
        processing_status[status_key] = {
            "article_id": article["id"],
            "article_url": article["url"],
            "started_at": time.time(),
        }

        logger.info("[%s] Scoring article %d: %s", scorer.name, article["id"], article["url"])

        content = article["content"]
        if not content:
            await save_impact(article["id"], 0.0)
            self._clear_model_status(status_key)
            return True

        loop = asyncio.get_event_loop()
        try:
            score = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    scorer.score,
                    content,
                    prompt_impact,
                ),
                timeout=300,
            )
            await save_impact(article["id"], score)
            logger.info("[%s] Article %d: %.4f", scorer.name, article["id"], score)
        except asyncio.TimeoutError:
            logger.warning("[%s] Timeout on article %d — skipping", scorer.name, article["id"])
            await save_impact(article["id"], 0.0)
        except Exception as e:
            logger.error("[%s] Failed on article %d: %s", scorer.name, article["id"], e)
            await save_impact(article["id"], 0.0)

        self._clear_model_status(status_key)
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
                    aliases.extend(
                        a.strip() for a in raw_aliases.split(",") if a.strip()
                    )
                ticker_aliases[symbol] = aliases

            self._ticker_aliases = ticker_aliases

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._company_scanner.update_aliases,
                ticker_aliases,
            )

            logger.info(
                "Aliases from Sentinel: %d tickers, %d aliases total",
                len(ticker_aliases),
                len(self._company_scanner.all_aliases),
            )
            return True
        except Exception:
            logger.exception("Failed to fetch aliases from Sentinel")
            return False

    async def scan_next_article(self) -> bool:
        """Find an article missing aliases, scan incrementally, record results.

        Returns True if an article was scanned, False if none available.
        """
        if self._company_scanner._nlp is None:
            return False

        if not self._company_scanner.ready:
            await self.fetch_aliases()
            if not self._company_scanner.ready:
                return False

        all_aliases = list(dict.fromkeys(self._company_scanner.all_aliases))
        if not all_aliases:
            return False

        alias_hash = hashlib.sha1(
            "\n".join(sorted(all_aliases)).encode("utf-8")
        ).hexdigest()
        cursor_id: int | None = None
        state_hash: str | None = None

        raw_state = await get_setting(COMPANY_SCAN_STATE_KEY)
        if raw_state:
            try:
                state = json.loads(raw_state)
                if isinstance(state, dict):
                    raw_last_id = state.get("last_id")
                    if raw_last_id is not None:
                        cursor_id = int(raw_last_id)
                    raw_hash = state.get("alias_hash")
                    if isinstance(raw_hash, str):
                        state_hash = raw_hash
            except (json.JSONDecodeError, TypeError, ValueError):
                cursor_id = None
                state_hash = None

        # Alias list changed: restart the scan walk from newest rows.
        if state_hash != alias_hash:
            cursor_id = None

        article = None
        missing: list[str] = []
        next_cursor = cursor_id
        wrapped = False
        for _ in range(COMPANY_SCAN_MAX_BATCHES):
            candidates = await get_company_scan_candidates(
                last_id=next_cursor, limit=COMPANY_SCAN_BATCH_SIZE
            )
            if not candidates:
                if next_cursor is not None and not wrapped:
                    next_cursor = None
                    wrapped = True
                    continue
                next_cursor = None
                break

            for candidate in candidates:
                next_cursor = candidate["id"]
                scanned_raw = candidate.get("scanned_aliases")
                scanned_aliases: list[str] = []
                if scanned_raw:
                    try:
                        parsed = json.loads(scanned_raw)
                        if isinstance(parsed, list):
                            scanned_aliases = [
                                v for v in parsed if isinstance(v, str)
                            ]
                    except (json.JSONDecodeError, TypeError):
                        scanned_aliases = []

                scanned_set = set(scanned_aliases)
                candidate_missing = [a for a in all_aliases if a not in scanned_set]
                if candidate_missing:
                    article = candidate
                    missing = candidate_missing
                    break
            if article:
                break

        await set_setting(
            COMPANY_SCAN_STATE_KEY,
            json.dumps({"last_id": next_cursor, "alias_hash": alias_hash}),
        )

        if not article:
            return False

        processing_status["company-scanner"] = {
            "article_id": article["id"],
            "article_url": article["url"],
            "started_at": time.time(),
        }

        logger.info(
            "[company-scanner] Scanning article %d (%d missing aliases): %s",
            article["id"],
            len(missing),
            article["url"],
        )

        content = article.get("content")
        matched_aliases: list[str] = []
        if content and missing:
            loop = asyncio.get_event_loop()
            try:
                matched_aliases = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        self._company_scanner.scan,
                        content,
                        missing,
                    ),
                    timeout=300,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[company-scanner] Timeout on article %d — skipping", article["id"]
                )
            except Exception as e:
                logger.error(
                    "[company-scanner] Failed on article %d: %s", article["id"], e
                )

        # Save company mentions for matched aliases
        matched_tickers = set()
        alias_to_ticker = self._company_scanner.alias_to_ticker
        for alias in matched_aliases:
            ticker = alias_to_ticker.get(alias)
            if ticker and ticker not in matched_tickers:
                matched_tickers.add(ticker)
                await save_company_mention(article["id"], ticker)

        if matched_tickers:
            logger.info(
                "[company-scanner] Article %d: matched %s",
                article["id"],
                matched_tickers,
            )

        # Mark article as scanned for the full current alias list
        await save_scanned_aliases(article["id"], all_aliases)

        self._clear_model_status("company-scanner")
        return True

    async def validate_next_company_match(self, model_name: str | None = None) -> bool:
        """Validate a company scanner candidate using NLI.

        Tests "This article is about {company}." against article content.
        Accepts if entailment >= validate_threshold setting, deletes the row otherwise.
        Returns True if a row was processed, False if none available.
        """
        if not self._loaded:
            return False

        row = await get_next_unvalidated_company_result()
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

        status_key = f"validate:{model_name}" if model_name else "validate:deberta-aux"
        processing_status[status_key] = {
            "article_id": article_id,
            "article_url": article_url,
            "started_at": time.time(),
        }

        logger.info("[validate] Checking %s (%s) for article %d", ticker, company_name, article_id)

        # Fetch the full article to check for stored content
        full_article = await get_article(article_id)
        if full_article and full_article.get("content"):
            content = full_article["content"]
        else:
            content, _ = await fetch_article_content(article_url)
        if not content:
            logger.warning(
                "[validate] Could not fetch content for article %d — rejecting",
                article_id,
            )
            await delete_company_result(article_id, ticker)
            self._clear_model_status(status_key)
            return True

        model = None
        if model_name:
            model = self._get_model(model_name)
        if not model:
            model = self._get_model("deberta-aux") or self._get_model("deberta")
        if not model:
            self._clear_model_status(status_key)
            return False

        loop = asyncio.get_event_loop()
        try:
            entail = await loop.run_in_executor(
                self._executor,
                model.validate_company,
                content,
                company_name,
            )
            validate_threshold = float(await get_setting("validate_threshold") or "0.5")
            if entail >= validate_threshold:
                await mark_company_result_validated(article_id, ticker)
                logger.info(
                    "[validate] ACCEPTED %s for article %d (entail=%.4f)",
                    ticker,
                    article_id,
                    entail,
                )
            else:
                await delete_company_result(article_id, ticker)
                logger.info(
                    "[validate] REJECTED %s for article %d (entail=%.4f)",
                    ticker,
                    article_id,
                    entail,
                )
        except Exception as e:
            logger.error(
                "[validate] Failed for %s article %d: %s", ticker, article_id, e
            )
            await delete_company_result(article_id, ticker)

        self._clear_model_status(status_key)
        return True

    async def classify_next_company_sentiment(
        self, model_name: str | None = None
    ) -> bool:
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

            status_key = f"company_sentiment:{model_name}" if model_name else "company_sentiment:deberta-aux"
            processing_status[status_key] = {
                "article_id": article_id,
                "article_url": article_url,
                "started_at": time.time(),
            }

            logger.info(
                "[company-sentiment] Scoring %s for article %d", ticker, article_id
            )

            # Fetch the full article to check for stored content
            full_article = await get_article(article_id)
            if full_article and full_article.get("content"):
                content = full_article["content"]
            else:
                content, _ = await fetch_article_content(article_url)
            if not content:
                logger.warning(
                    "[company-sentiment] Could not fetch content for article %d",
                    article_id,
                )
                await save_company_sentiment(article_id, ticker, 0.0, 0.0)
                self._clear_model_status(status_key)
                return True

            model = None
            if model_name:
                model = self._get_model(model_name)
            if not model:
                model = self._get_model("deberta-aux") or self._get_model("deberta")
            if not model:
                self._clear_model_status(status_key)
                return False

            loop = asyncio.get_event_loop()
            try:
                sentiment, impact = await loop.run_in_executor(
                    self._executor,
                    model.score_company_sentiment,
                    content,
                    company_name,
                    prompt_company,
                )
                await save_company_sentiment(article_id, ticker, sentiment, impact)
                logger.info(
                    "[company-sentiment] %s article %d: sentiment=%.4f impact=%.4f",
                    ticker,
                    article_id,
                    sentiment,
                    impact,
                )
            except Exception as e:
                logger.error(
                    "[company-sentiment] Failed for %s article %d: %s",
                    ticker,
                    article_id,
                    e,
                )
                await save_company_sentiment(article_id, ticker, 0.0, 0.0)

            self._clear_model_status(status_key)
            return True

    async def _prefetch_content_loop(self):
        """Pre-fetch article content into DB so workers never wait for HTTP."""
        logger.info("[prefetcher] Started")
        try:
            while _workers_running:
                articles = await get_articles_needing_content(limit=5)
                if not articles:
                    await asyncio.sleep(5)
                    continue
                tasks = [self._fetch_and_store(a) for a in articles]
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        logger.info("[prefetcher] Stopped")

    async def _fetch_and_store(self, article):
        """Fetch content for a single article and store it in the DB."""
        try:
            content, published_at = await fetch_article_content(article["url"])
            has_content = bool(content and content.strip())
            existing_published_at = article.get("published_at")
            resolved_published_at = (
                published_at if published_at is not None else existing_published_at
            )

            if not has_content or resolved_published_at is None:
                await delete_article(article["id"])
                return

            await store_article_content(article["id"], content, published_at)
        except Exception as e:
            logger.error("[prefetcher] Failed for article %d: %s", article["id"], e)
            await delete_article(article["id"])

    def _get_model(self, name: str) -> BaseModel | None:
        for m in self._models:
            if m.name == name:
                return m
        return None

    def _clear_model_status(self, model_name: str):
        status = processing_status.get(model_name, {})
        started = status.get("started_at")
        if started:
            duration = time.time() - started
            if model_name not in worker_durations:
                worker_durations[model_name] = deque(maxlen=10)
            worker_durations[model_name].append(duration)
        processing_status[model_name] = {
            "article_id": None,
            "article_url": None,
            "started_at": None,
        }

    async def _worker_loop(
        self, worker_name: str, capabilities: list[tuple[str, str | None]]
    ):
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
                            did_work = await self.score_next_impact(model)
                        elif task_type == "scan":
                            did_work = await self.scan_next_article()
                        elif task_type == "validate":
                            did_work = await self.validate_next_company_match(
                                model_name=model
                            )
                        elif task_type == "company_sentiment":
                            did_work = await self.classify_next_company_sentiment(
                                model_name=model
                            )
                    except Exception:
                        logger.exception(
                            "[worker:%s] Error in %s", worker_name, task_type
                        )

                await _thermal_gate()
                if not did_work:
                    await asyncio.sleep(IDLE_SLEEP_NONE)
        except asyncio.CancelledError:
            pass

        logger.info("[worker:%s] Stopped", worker_name)


# Singleton
ensemble = EnsembleClassifier()


def get_worker_avg_times() -> dict[str, float | None]:
    """Return average inference time (seconds) per worker, or None if no data."""
    result = {}
    for name in processing_status:
        durations = worker_durations.get(name)
        if durations:
            result[name] = round(sum(durations) / len(durations), 1)
        else:
            result[name] = None
    return result


def start_workers():
    """Launch all pull-based worker loops as async tasks."""
    global _workers_running
    _workers_running = True
    # Launch content pre-fetcher
    prefetch_task = asyncio.create_task(
        ensemble._prefetch_content_loop(),
        name="prefetcher",
    )
    _worker_tasks.append(prefetch_task)
    for worker_name, capabilities in WORKER_CONFIGS:
        task = asyncio.create_task(
            ensemble._worker_loop(worker_name, capabilities),
            name=f"worker:{worker_name}",
        )
        _worker_tasks.append(task)
    logger.info("Started %d workers + prefetcher", len(_worker_tasks) - 1)


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
