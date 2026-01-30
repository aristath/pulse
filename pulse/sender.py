import json
import logging
import httpx

from pulse.database import get_unsent_articles, get_results_for_article, mark_sent

logger = logging.getLogger(__name__)

SENTINEL_URL = "http://localhost:8000"  # TODO: configure via env


async def send_to_sentinel():
    """Send processed articles to Sentinel via REST API."""
    articles = await get_unsent_articles()
    if not articles:
        return

    sent = 0
    async with httpx.AsyncClient(timeout=30) as client:
        for article in articles:
            try:
                results = await get_results_for_article(article["id"])
                if not results:
                    continue

                payload = {
                    "article_id": article["id"],
                    "url": article["url"],
                    "models": {
                        r["model"]: r["signals"] for r in results
                    },
                }

                resp = await client.post(
                    f"{SENTINEL_URL}/api/pulse/signals",
                    json=payload,
                )
                resp.raise_for_status()
                await mark_sent(article["id"])
                sent += 1
            except Exception:
                logger.exception("Failed to send article %d to Sentinel", article["id"])

    if sent:
        logger.info("Sent %d articles to Sentinel", sent)
