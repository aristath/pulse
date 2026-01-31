import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pulse.fetcher import fetch_all_feeds
from pulse.classifier import ensemble
from pulse.fundus_crawler import crawl_enabled_publishers
from pulse.database import clear_processed_content

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _fetch_job():
    """Scheduled job: fetch all RSS feeds."""
    logger.info("Running RSS fetch job")
    try:
        await fetch_all_feeds()
    except Exception:
        logger.exception("RSS fetch job failed")


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


async def _fundus_crawl_job():
    """Scheduled job: crawl enabled fundus publishers."""
    logger.info("Running fundus crawl job")
    try:
        await crawl_enabled_publishers()
    except Exception:
        logger.exception("Fundus crawl job failed")


async def _cleanup_content_job():
    """Scheduled job: clear content for fully processed articles."""
    try:
        model_count = max(len(ensemble.model_names), 1)
        await clear_processed_content(model_count)
    except Exception:
        logger.exception("Content cleanup job failed")


def start_scheduler():
    """Configure and start the APScheduler."""
    scheduler.add_job(
        _fetch_job, "interval", minutes=5, id="fetch_rss", name="Fetch RSS feeds"
    )
    scheduler.add_job(
        _refresh_labels_job,
        "interval",
        minutes=30,
        id="refresh_labels",
        name="Refresh labels",
    )
    scheduler.add_job(
        _refresh_aliases_job,
        "interval",
        minutes=30,
        id="refresh_aliases",
        name="Refresh aliases",
    )
    scheduler.add_job(
        _fundus_crawl_job,
        "interval",
        minutes=10,
        id="fundus_crawl",
        name="Fundus crawl",
    )
    scheduler.add_job(
        _cleanup_content_job,
        "interval",
        minutes=30,
        id="cleanup_content",
        name="Content cleanup",
    )
    scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler():
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")
