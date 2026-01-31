import asyncio
import json
import logging
from datetime import timezone

from fundus import Crawler, PublisherCollection, RSSFeed, NewsMap
from fundus.publishers.base_objects import Publisher, PublisherGroup

from pulse.database import get_setting, add_fundus_articles

logger = logging.getLogger(__name__)

# Country codes used by PublisherCollection
_COUNTRY_CODES = [
    attr
    for attr in dir(PublisherCollection)
    if not attr.startswith("_")
    and isinstance(getattr(PublisherCollection, attr), PublisherGroup)
]


def get_available_publishers() -> dict:
    """Enumerate all fundus publishers, grouped by country code.

    Returns dict keyed by country code (e.g. "us") with list of dicts:
    {id, name, domain}
    """
    result = {}
    for code in _COUNTRY_CODES:
        group = getattr(PublisherCollection, code)
        publishers = []
        for attr in dir(group):
            obj = getattr(group, attr, None)
            if isinstance(obj, Publisher):
                publishers.append(
                    {
                        "id": f"{code}.{attr}",
                        "name": obj.name,
                        "domain": obj.domain,
                    }
                )
        if publishers:
            publishers.sort(key=lambda p: p["name"])
            result[code] = publishers
    return result


async def get_enabled_publisher_ids() -> list[str]:
    """Read enabled fundus publisher IDs from the settings table."""
    raw = await get_setting("fundus_publishers")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def resolve_publishers(ids: list[str]) -> list[Publisher]:
    """Convert identifier strings (e.g. 'us.CNBC') back to Publisher objects."""
    publishers = []
    for pid in ids:
        parts = pid.split(".", 1)
        if len(parts) != 2:
            logger.warning("Invalid publisher id: %s", pid)
            continue
        country, attr = parts
        group = getattr(PublisherCollection, country, None)
        if group is None:
            logger.warning("Unknown country code: %s", country)
            continue
        pub = getattr(group, attr, None)
        if not isinstance(pub, Publisher):
            logger.warning("Unknown publisher: %s", pid)
            continue
        publishers.append(pub)
    return publishers


def _crawl_sync(publishers: list[Publisher], timeout: int = 120) -> list[dict]:
    """Run fundus Crawler synchronously (called via asyncio.to_thread).

    Returns list of article dicts ready for DB insertion.
    """
    crawler = Crawler(*publishers, restrict_sources_to=[RSSFeed, NewsMap])
    articles = []
    for article in crawler.crawl(timeout=timeout):
        try:
            url = article.html.requested_url
            title = article.title or ""
            body = str(article.body) if article.body else ""
            pub_date = article.publishing_date

            if not url or not body:
                continue

            # Prepend title to content (matching fetch_article_content behavior)
            content = f"{title}\n\n{body}" if title else body

            published_at = None
            if pub_date:
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                published_at = pub_date.isoformat()

            articles.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "published_at": published_at,
                }
            )
        except Exception:
            logger.exception("Error processing fundus article")
            continue

    return articles


async def crawl_enabled_publishers() -> int:
    """Main crawl function: read enabled publishers, crawl, insert into DB.

    Returns count of newly inserted articles.
    """
    ids = await get_enabled_publisher_ids()
    if not ids:
        logger.info("No fundus publishers enabled")
        return 0

    publishers = resolve_publishers(ids)
    if not publishers:
        logger.warning("No valid fundus publishers resolved from: %s", ids)
        return 0

    logger.info("Crawling %d fundus publishers", len(publishers))

    articles = await asyncio.to_thread(_crawl_sync, publishers)

    if not articles:
        logger.info("Fundus crawl returned no articles")
        return 0

    added = await add_fundus_articles(articles)
    logger.info("Fundus crawl: %d new / %d total articles", added, len(articles))
    return added
