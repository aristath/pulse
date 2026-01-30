import logging
import time
from datetime import datetime, timezone

import feedparser
import httpx
from readability import Document

from pulse.database import get_feeds, add_articles

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Pulse/0.1 (RSS Signal Generator)"
}


async def fetch_all_feeds():
    """Fetch all enabled RSS feeds and save new article URLs."""
    feeds = await get_feeds(enabled_only=True)
    if not feeds:
        logger.info("No enabled feeds to fetch")
        return

    total_new = 0
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for feed in feeds:
            try:
                new = await _fetch_feed(client, feed)
                total_new += new
            except Exception:
                logger.exception("Failed to fetch feed %s", feed["url"])

    logger.info("Fetched %d new articles from %d feeds", total_new, len(feeds))


async def _fetch_feed(client: httpx.AsyncClient, feed: dict) -> int:
    """Fetch a single RSS feed and save new article URLs. Returns count of new articles."""
    resp = await client.get(feed["url"])
    resp.raise_for_status()

    parsed = feedparser.parse(resp.text)
    articles = []
    for entry in parsed.entries:
        url = entry.get("link")
        if url:
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
            articles.append({"url": url, "published_at": published_at})

    if articles:
        new = await add_articles(feed["id"], articles)
        logger.info("Feed '%s': %d new / %d total entries", feed.get("name") or feed["url"], new, len(articles))
        return new
    return 0


_content_cache: dict[str, tuple[float, str | None, str | None]] = {}
_CACHE_TTL = 28800  # 8 hours


def _cache_cleanup():
    """Remove expired entries."""
    now = time.time()
    expired = [k for k, (ts, _, _) in _content_cache.items() if now - ts > _CACHE_TTL]
    for k in expired:
        del _content_cache[k]


async def fetch_article_content(url: str) -> tuple[str | None, str | None]:
    """Fetch and extract readable content from an article URL.

    Returns (text_content, published_at_iso) tuple. Results are cached in-memory.
    """
    if url in _content_cache:
        ts, text, pub = _content_cache[url]
        if time.time() - ts < _CACHE_TTL:
            return text, pub
        del _content_cache[url]

    # Periodic cleanup
    if len(_content_cache) > 100:
        _cache_cleanup()

    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            doc = Document(html)
            title = doc.title()
            content = doc.summary()
            # Strip HTML tags for plain text
            from lxml.html import fromstring
            tree = fromstring(content)
            text = tree.text_content().strip()
            if title:
                text = f"{title}\n\n{text}"

            # Try to extract published date from HTML meta tags
            published_at = _extract_date_from_html(html)

            _content_cache[url] = (time.time(), text, published_at)
            return text, published_at
    except Exception:
        logger.exception("Failed to fetch article content: %s", url)
        return None, None



def _extract_date_from_html(html: str) -> str | None:
    """Extract published date from common HTML meta tags."""
    import re
    # Common meta tag patterns for article dates
    patterns = [
        r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"',
        r'<meta[^>]+content="([^"]+)"[^>]+property="article:published_time"',
        r'<meta[^>]+name="publish-date"[^>]+content="([^"]+)"',
        r'<meta[^>]+name="date"[^>]+content="([^"]+)"',
        r'"datePublished"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            raw = match.group(1).strip()
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return dt.isoformat()
            except ValueError:
                return raw
    return None
