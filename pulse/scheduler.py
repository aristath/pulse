import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pulse.fetcher import fetch_all_feeds
from pulse.classifier import ensemble

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()
_model_running: set[str] = set()
_impact_running = False
_scanning_running = False
_company_sentiment_running = False


async def _fetch_job():
    """Scheduled job: fetch all RSS feeds."""
    logger.info("Running RSS fetch job")
    try:
        await fetch_all_feeds()
    except Exception:
        logger.exception("RSS fetch job failed")


async def _classify_job():
    """Scheduled job: kick off classification for each idle model."""
    if not ensemble._loaded:
        return
    for model_name in ensemble.model_names:
        if model_name not in _model_running:
            _model_running.add(model_name)
            asyncio.create_task(_run_model_classify(model_name))


async def _run_model_classify(model_name: str):
    """Run one classify cycle for a single model."""
    try:
        processed = await ensemble.classify_next_for_model(model_name)
        if not processed:
            logger.debug("No unprocessed articles for %s", model_name)
    except Exception:
        logger.exception("Classify failed for model %s", model_name)
    finally:
        _model_running.discard(model_name)


async def _impact_job():
    """Scheduled job: score next article for impact."""
    global _impact_running
    if not ensemble._loaded or _impact_running:
        return
    _impact_running = True
    try:
        processed = await ensemble.score_next_impact()
        if not processed:
            logger.debug("No unscored articles for impact")
    except Exception:
        logger.exception("Impact scoring failed")
    finally:
        _impact_running = False


async def _refresh_labels_job():
    """Scheduled job: refresh labels from Sentinel."""
    try:
        await ensemble.fetch_labels()
    except Exception:
        logger.exception("Label refresh failed")


async def _refresh_aliases_job():
    """Scheduled job: refresh company aliases from Sentinel."""
    try:
        await ensemble.fetch_aliases()
    except Exception:
        logger.exception("Alias refresh failed")


async def _scan_job():
    """Scheduled job: scan next article for company mentions."""
    global _scanning_running
    if not ensemble._loaded or _scanning_running:
        return
    _scanning_running = True
    try:
        processed = await ensemble.scan_next_article()
        if not processed:
            logger.debug("No unscanned articles for company scanner")
    except Exception:
        logger.exception("Company scan failed")
    finally:
        _scanning_running = False


async def _company_sentiment_job():
    """Scheduled job: classify sentiment for next unscored company mention."""
    global _company_sentiment_running
    if not ensemble._loaded or _company_sentiment_running:
        return
    _company_sentiment_running = True
    try:
        processed = await ensemble.classify_next_company_sentiment()
        if not processed:
            logger.debug("No unscored company results")
    except Exception:
        logger.exception("Company sentiment classification failed")
    finally:
        _company_sentiment_running = False


def start_scheduler():
    """Configure and start the APScheduler."""
    scheduler.add_job(_fetch_job, "interval", minutes=5, id="fetch_rss", name="Fetch RSS feeds")
    scheduler.add_job(_impact_job, "interval", seconds=5, id="impact", name="Score article impact")
    scheduler.add_job(_classify_job, "interval", seconds=5, id="classify", name="Classify articles")
    scheduler.add_job(_refresh_labels_job, "interval", minutes=30, id="refresh_labels", name="Refresh labels")
    scheduler.add_job(_refresh_aliases_job, "interval", minutes=30, id="refresh_aliases", name="Refresh aliases")
    scheduler.add_job(_scan_job, "interval", seconds=5, id="scan_companies", name="Scan company mentions")
    scheduler.add_job(_company_sentiment_job, "interval", seconds=5, id="company_sentiment", name="Company sentiment")
    scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler():
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")
