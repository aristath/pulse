import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import aiosqlite
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)


def _normalize_date(value) -> int | None:
    """Convert any date representation to a unix timestamp (int)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    try:
        return int(dateutil_parser.parse(str(value)).timestamp())
    except (ValueError, TypeError):
        logger.warning("Could not parse date: %r", value)
        return None


DB_PATH = Path(__file__).parent.parent / "data" / "pulse.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    name TEXT,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    feed_id INTEGER REFERENCES feeds(id),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT,
    published_at TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sent_to_sentinel BOOLEAN DEFAULT 0,
    sent_at TIMESTAMP,
    impact REAL,
    scanned_aliases JSON
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    model TEXT NOT NULL,
    signals TEXT NOT NULL,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, model)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS company_results (
    id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    ticker TEXT NOT NULL,
    validated BOOLEAN,
    sentiment REAL,
    impact REAL,
    classified_at TIMESTAMP,
    UNIQUE(article_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_results_article ON results(article_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_company_results_article ON company_results(article_id);
"""


async def get_db() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
        # Migration: add columns if missing (existing DBs)
        cursor = await db.execute("PRAGMA table_info(articles)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "impact" not in columns:
            await db.execute("ALTER TABLE articles ADD COLUMN impact REAL")
        if "title" not in columns:
            await db.execute("ALTER TABLE articles ADD COLUMN title TEXT")
        if "content" not in columns:
            await db.execute("ALTER TABLE articles ADD COLUMN content TEXT")
        # Migration: add scanned_aliases column if missing (existing DBs)
        if "scanned_aliases" not in columns:
            await db.execute("ALTER TABLE articles ADD COLUMN scanned_aliases JSON")
        # Migration: drop legacy alias_scans table
        await db.execute("DROP TABLE IF EXISTS alias_scans")
        # Migration: add validated column if missing (existing DBs)
        cursor = await db.execute("PRAGMA table_info(company_results)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "validated" not in columns:
            await db.execute("ALTER TABLE company_results ADD COLUMN validated BOOLEAN")
        await db.commit()
    finally:
        await db.close()


# --- Feeds ---


async def get_feeds(enabled_only: bool = False) -> list[dict]:
    db = await get_db()
    try:
        sql = "SELECT * FROM feeds"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY created_at DESC"
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def add_feed(url: str, name: str | None = None) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO feeds (url, name) VALUES (?, ?)",
            (url, name),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def delete_feed(feed_id: int):
    db = await get_db()
    try:
        await db.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
        await db.commit()
    finally:
        await db.close()


async def update_feed(feed_id: int, **kwargs):
    db = await get_db()
    try:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [feed_id]
        await db.execute(f"UPDATE feeds SET {sets} WHERE id = ?", vals)
        await db.commit()
    finally:
        await db.close()


# --- Articles ---


async def add_article_url(url: str) -> int | None:
    """Insert a single article URL (no feed). Returns article id, or None if duplicate."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO articles (feed_id, url) VALUES (NULL, ?)", (url,)
        )
        await db.commit()
        return cursor.lastrowid
    except aiosqlite.IntegrityError:
        return None
    finally:
        await db.close()


async def add_articles(feed_id: int, articles: list[dict]) -> int:
    """Insert articles, ignore duplicates. Returns count of new articles."""
    db = await get_db()
    try:
        added = 0
        for article in articles:
            try:
                await db.execute(
                    "INSERT INTO articles (feed_id, url, published_at) VALUES (?, ?, ?)",
                    (feed_id, article["url"], _normalize_date(article.get("published_at"))),
                )
                added += 1
            except aiosqlite.IntegrityError:
                pass
        await db.commit()
        return added
    finally:
        await db.close()


async def get_unprocessed_article_for_model(model: str) -> dict | None:
    """Get an article that hasn't been classified by this model.

    Only considers articles above the classify impact threshold, ordered by highest impact first.
    """
    threshold = float(await get_setting("classify_threshold") or "0.5")
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT a.* FROM articles a
            LEFT JOIN results r ON a.id = r.article_id AND r.model = ?
            WHERE r.id IS NULL
              AND a.impact IS NOT NULL
              AND a.impact >= ?
              AND a.content IS NOT NULL AND a.content != ''
            ORDER BY a.impact DESC
            LIMIT 1
        """,
            (model, threshold),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def get_next_article_for_impact() -> dict | None:
    """Claim and return an article that hasn't been impact-scored yet, newest first.

    Sets impact = -1.0 as a claim sentinel so parallel workers don't pick
    the same row. The real score overwrites this after inference.
    """
    db = await get_db()
    try:
        cursor = await db.execute("""
            UPDATE articles
            SET impact = -1.0
            WHERE id = (
                SELECT a.id FROM articles a
                WHERE a.impact IS NULL
                  AND a.content IS NOT NULL AND a.content != ''
                ORDER BY a.published_at DESC, a.id DESC
                LIMIT 1
            )
            RETURNING *
        """)
        row = await cursor.fetchone()
        await db.commit()
        return dict(row) if row else None
    finally:
        await db.close()


async def save_impact(article_id: int, impact: float):
    """Save the impact score for an article."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE articles SET impact = ? WHERE id = ?",
            (impact, article_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_articles(
    limit: int = 50,
    offset: int = 0,
    processed_only: bool = False,
) -> list[dict]:
    db = await get_db()
    try:
        if processed_only:
            sql = """
                SELECT DISTINCT a.* FROM articles a
                INNER JOIN results r ON a.id = r.article_id
                ORDER BY a.fetched_at DESC LIMIT ? OFFSET ?
            """
        else:
            sql = "SELECT * FROM articles ORDER BY fetched_at DESC LIMIT ? OFFSET ?"
        cursor = await db.execute(sql, (limit, offset))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def update_article_date(article_id: int, published_at: str):
    db = await get_db()
    try:
        await db.execute(
            "UPDATE articles SET published_at = ? WHERE id = ? AND published_at IS NULL",
            (_normalize_date(published_at), article_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_article(article_id: int) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# --- Results ---


async def save_result(article_id: int, model: str, signals: dict):
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO results (article_id, model, signals) VALUES (?, ?, ?)",
            (article_id, model, json.dumps(signals)),
        )
        await db.commit()
    finally:
        await db.close()


async def get_results_for_article(article_id: int) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM results WHERE article_id = ? ORDER BY model",
            (article_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["signals"] = json.loads(d["signals"])
            results.append(d)
        return results
    finally:
        await db.close()


async def mark_sent(article_id: int):
    db = await get_db()
    try:
        await db.execute(
            "UPDATE articles SET sent_to_sentinel = 1, sent_at = CURRENT_TIMESTAMP WHERE id = ?",
            (article_id,),
        )
        await db.commit()
    finally:
        await db.close()


async def get_unsent_articles() -> list[dict]:
    """Get articles that have results but haven't been sent to Sentinel."""
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT DISTINCT a.* FROM articles a
            INNER JOIN results r ON a.id = r.article_id
            WHERE a.sent_to_sentinel = 0
            ORDER BY a.fetched_at ASC
        """)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# --- Stats ---


async def get_stats(model_count: int = 1) -> dict:
    db = await get_db()
    try:
        total = (await (await db.execute("SELECT COUNT(*) FROM articles")).fetchone())[
            0
        ]
        processed = (
            await (
                await db.execute(
                    """
            SELECT COUNT(*) FROM (
                SELECT article_id FROM results
                GROUP BY article_id
                HAVING COUNT(DISTINCT model) >= ?
            )
        """,
                    (model_count,),
                )
            ).fetchone()
        )[0]
        sent = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE sent_to_sentinel = 1"
                )
            ).fetchone()
        )[0]
        feeds = (await (await db.execute("SELECT COUNT(*) FROM feeds")).fetchone())[0]

        # Per-model classify counts
        cursor = await db.execute(
            "SELECT model, COUNT(*) as cnt FROM results GROUP BY model"
        )
        classify_counts = {row[0]: row[1] for row in await cursor.fetchall()}

        # Impact stats
        impact_scored = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE impact IS NOT NULL"
                )
            ).fetchone()
        )[0]
        classify_threshold = float(await get_setting("classify_threshold") or "0.5")
        impact_relevant = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE impact IS NOT NULL AND impact >= ?",
                    (classify_threshold,),
                )
            ).fetchone()
        )[0]

        # Company scan stats
        scan_threshold = float(await get_setting("scan_threshold") or "0.3")
        scan_eligible = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE impact IS NOT NULL AND impact >= ?",
                    (scan_threshold,),
                )
            ).fetchone()
        )[0]
        # "Fully scanned" = scanned_aliases has the max alias count
        max_alias_count = (
            await (
                await db.execute(
                    "SELECT MAX(json_array_length(scanned_aliases)) FROM articles WHERE scanned_aliases IS NOT NULL"
                )
            ).fetchone()
        )[0] or 0
        company_scanned = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE scanned_aliases IS NOT NULL AND json_array_length(scanned_aliases) >= ?",
                    (max_alias_count,),
                )
            ).fetchone()
        )[0] if max_alias_count > 0 else 0
        company_mentions = (
            await (await db.execute("SELECT COUNT(*) FROM company_results")).fetchone()
        )[0]
        company_validated = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM company_results WHERE validated = 1"
                )
            ).fetchone()
        )[0]
        company_scored = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM company_results WHERE sentiment IS NOT NULL"
                )
            ).fetchone()
        )[0]

        # Build per-worker progress: (done, total) keyed by processing_status keys
        per_worker = {
            "impact:impact-1": (impact_scored, total),
            "impact:impact-2": (impact_scored, total),
            "impact:impact-3": (impact_scored, total),
        }
        for model_name, count in classify_counts.items():
            per_worker[f"classify:{model_name}"] = (count, impact_relevant)
        per_worker["company-scanner"] = (company_scanned, scan_eligible)
        # Validate/sentiment are shared across models — show on first available
        per_worker["validate:deberta-aux"] = (company_validated, company_mentions)
        per_worker["company_sentiment:deberta-aux"] = (company_scored, company_validated)

        return {
            "total_articles": total,
            "processed_articles": processed,
            "unprocessed_articles": total - processed,
            "sent_articles": sent,
            "total_feeds": feeds,
            "per_model": classify_counts,
            "per_worker": per_worker,
            "impact_scored": impact_scored,
            "impact_relevant": impact_relevant,
            "impact_unscored": total - impact_scored,
            "company_scanned": company_scanned,
            "company_mentions": company_mentions,
            "company_scored": company_scored,
        }
    finally:
        await db.close()


# --- Charts ---


async def get_sentiment_bars(bar_type: str, threshold: float, days: int = 90, decay: bool = True) -> list[dict]:
    """Return (optionally decay-weighted) average sentiment for the last N days.

    When decay=True: weight = exp(-0.1 * age_days).
    When decay=False: uniform weight (simple average).
    Returns [{label, avg_sentiment, article_count}, ...] sorted by avg_sentiment DESC.
    """
    now = time.time()
    cutoff = now - days * 86400
    db = await get_db()
    try:
        if bar_type == "country":
            cursor = await db.execute(
                """
                SELECT r.article_id, country.key AS label,
                       AVG(CAST(sector.value AS REAL)) AS sentiment,
                       a.published_at
                FROM results r
                JOIN articles a ON r.article_id = a.id,
                     json_each(r.signals) AS country,
                     json_each(country.value) AS sector
                WHERE a.published_at >= ?
                  AND a.impact IS NOT NULL AND a.impact >= ?
                  AND r.signals != '{}'
                GROUP BY r.article_id, country.key
                """,
                (cutoff, threshold),
            )
        elif bar_type == "industry":
            cursor = await db.execute(
                """
                SELECT r.article_id, sector.key AS label,
                       AVG(CAST(sector.value AS REAL)) AS sentiment,
                       a.published_at
                FROM results r
                JOIN articles a ON r.article_id = a.id,
                     json_each(r.signals) AS country,
                     json_each(country.value) AS sector
                WHERE a.published_at >= ?
                  AND a.impact IS NOT NULL AND a.impact >= ?
                  AND r.signals != '{}'
                GROUP BY r.article_id, sector.key
                """,
                (cutoff, threshold),
            )
        elif bar_type == "detailed":
            cursor = await db.execute(
                """
                SELECT r.article_id, country.key AS country,
                       sector.key AS industry,
                       CAST(sector.value AS REAL) AS sentiment,
                       a.published_at
                FROM results r
                JOIN articles a ON r.article_id = a.id,
                     json_each(r.signals) AS country,
                     json_each(country.value) AS sector
                WHERE a.published_at >= ?
                  AND a.impact IS NOT NULL AND a.impact >= ?
                  AND r.signals != '{}'
                """,
                (cutoff, threshold),
            )
        elif bar_type == "company":
            cursor = await db.execute(
                """
                SELECT cr.article_id, cr.ticker AS label,
                       cr.sentiment, a.published_at
                FROM company_results cr
                JOIN articles a ON cr.article_id = a.id
                WHERE a.published_at >= ?
                  AND cr.sentiment IS NOT NULL
                """,
                (cutoff,),
            )
        else:
            raise ValueError(f"bar_type must be 'country', 'industry', 'company', or 'detailed', got {bar_type!r}")

        rows = await cursor.fetchall()
    finally:
        await db.close()

    if bar_type == "detailed":
        groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for row in rows:
            pub = row["published_at"]
            if pub is None:
                continue
            age_days = (now - float(pub)) / 86400
            weight = math.exp(-0.1 * age_days) if decay else 1.0
            key = (row["country"], row["industry"])
            groups.setdefault(key, []).append((row["sentiment"], weight))

        result = []
        for (country, industry), entries in groups.items():
            total_weight = sum(w for _, w in entries)
            if total_weight == 0:
                continue
            avg = sum(s * w for s, w in entries) / total_weight
            result.append({
                "country": country,
                "industry": industry,
                "avg_sentiment": avg,
                "article_count": len(entries),
            })
        result.sort(key=lambda x: x["avg_sentiment"], reverse=True)
        return result

    # Group by label and compute decay-weighted average
    label_groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        label = row["label"]
        pub = row["published_at"]
        if pub is None:
            continue
        age_days = (now - float(pub)) / 86400
        weight = math.exp(-0.1 * age_days) if decay else 1.0
        label_groups.setdefault(label, []).append((row["sentiment"], weight))

    result = []
    for label, entries in label_groups.items():
        total_weight = sum(w for _, w in entries)
        if total_weight == 0:
            continue
        avg = sum(s * w for s, w in entries) / total_weight
        result.append({
            "label": label,
            "avg_sentiment": avg,
            "article_count": len(entries),
        })
    result.sort(key=lambda x: x["avg_sentiment"], reverse=True)
    return result


async def get_sentiment_bar_articles(
    bar_type: str, label: str, threshold: float, days: int = 90, decay: bool = True
) -> list[dict]:
    """Return individual articles contributing to a sentiment bar, ordered by contribution.

    Each entry has: title, url, sentiment, weight, contribution (sentiment * weight).
    """
    now = time.time()
    cutoff = now - days * 86400
    db = await get_db()
    try:
        if bar_type == "country":
            cursor = await db.execute(
                """
                SELECT r.article_id, a.title, a.url,
                       AVG(CAST(sector.value AS REAL)) AS sentiment,
                       a.published_at
                FROM results r
                JOIN articles a ON r.article_id = a.id,
                     json_each(r.signals) AS country,
                     json_each(country.value) AS sector
                WHERE a.published_at >= ?
                  AND a.impact IS NOT NULL AND a.impact >= ?
                  AND r.signals != '{}'
                  AND country.key = ?
                GROUP BY r.article_id
                """,
                (cutoff, threshold, label),
            )
        elif bar_type == "industry":
            cursor = await db.execute(
                """
                SELECT r.article_id, a.title, a.url,
                       AVG(CAST(sector.value AS REAL)) AS sentiment,
                       a.published_at
                FROM results r
                JOIN articles a ON r.article_id = a.id,
                     json_each(r.signals) AS country,
                     json_each(country.value) AS sector
                WHERE a.published_at >= ?
                  AND a.impact IS NOT NULL AND a.impact >= ?
                  AND r.signals != '{}'
                  AND sector.key = ?
                GROUP BY r.article_id
                """,
                (cutoff, threshold, label),
            )
        elif bar_type == "company":
            cursor = await db.execute(
                """
                SELECT cr.article_id, a.title, a.url,
                       cr.sentiment, a.published_at
                FROM company_results cr
                JOIN articles a ON cr.article_id = a.id
                WHERE a.published_at >= ?
                  AND cr.sentiment IS NOT NULL
                  AND cr.ticker = ?
                """,
                (cutoff, label),
            )
        else:
            return []

        rows = await cursor.fetchall()
    finally:
        await db.close()

    result = []
    for row in rows:
        pub = row["published_at"]
        if pub is None:
            continue
        age_days = (now - float(pub)) / 86400
        weight = math.exp(-0.1 * age_days) if decay else 1.0
        result.append({
            "title": row["title"] or row["url"],
            "url": row["url"],
            "sentiment": row["sentiment"],
            "weight": round(weight, 4),
            "contribution": abs(row["sentiment"] * weight),
        })
    result.sort(key=lambda x: x["contribution"], reverse=True)
    return result


async def get_articles_by_month() -> list[dict]:
    """Return count of relevant articles per month, newest first.

    An article is 'relevant' if any model produced non-empty signals for it.
    """
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """
            SELECT strftime('%Y-%m', a.published_at, 'unixepoch') AS month,
                   COUNT(DISTINCT r.article_id) AS cnt
            FROM results r
            JOIN articles a ON r.article_id = a.id
            WHERE r.signals != '{}'
              AND a.published_at IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
            """,
        )
        return [{"month": row[0], "count": row[1]} for row in rows]
    finally:
        await db.close()


# --- Settings ---

# --- Alias Scans & Company Results ---


async def get_article_missing_alias(alias: str) -> dict | None:
    """Get an article that hasn't been scanned for this alias, above impact threshold, highest impact first."""
    threshold = float(await get_setting("scan_threshold") or "0.3")
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT a.* FROM articles a
            WHERE a.impact IS NOT NULL AND a.impact >= ?
              AND a.content IS NOT NULL AND a.content != ''
              AND (a.scanned_aliases IS NULL
                   OR ? NOT IN (SELECT value FROM json_each(a.scanned_aliases)))
            ORDER BY a.impact DESC
            LIMIT 1
        """,
            (threshold, alias),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def get_scanned_aliases(article_id: int) -> list[str]:
    """Return the list of aliases already scanned for this article."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT scanned_aliases FROM articles WHERE id = ?", (article_id,)
        )
        row = await cursor.fetchone()
        if row and row["scanned_aliases"]:
            return json.loads(row["scanned_aliases"])
        return []
    finally:
        await db.close()


async def save_scanned_aliases(article_id: int, aliases: list[str]):
    """Store the full list of scanned aliases for an article."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE articles SET scanned_aliases = ? WHERE id = ?",
            (json.dumps(aliases), article_id),
        )
        await db.commit()
    finally:
        await db.close()


async def save_company_mention(article_id: int, ticker: str):
    """Record a company mention (with NULL sentiment until scored)."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR IGNORE INTO company_results (article_id, ticker) VALUES (?, ?)",
            (article_id, ticker),
        )
        await db.commit()
    finally:
        await db.close()


async def get_next_unvalidated_company_result() -> dict | None:
    """Get a company_results row where validated is NULL (not yet checked by NLI)."""
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT cr.*, a.url as article_url
            FROM company_results cr
            JOIN articles a ON cr.article_id = a.id
            WHERE cr.validated IS NULL
            ORDER BY a.impact DESC
            LIMIT 1
        """)
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def mark_company_result_validated(article_id: int, ticker: str):
    """Mark a company_results row as validated."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE company_results SET validated = 1 WHERE article_id = ? AND ticker = ?",
            (article_id, ticker),
        )
        await db.commit()
    finally:
        await db.close()


async def delete_company_result(article_id: int, ticker: str):
    """Delete a rejected company_results row."""
    db = await get_db()
    try:
        await db.execute(
            "DELETE FROM company_results WHERE article_id = ? AND ticker = ?",
            (article_id, ticker),
        )
        await db.commit()
    finally:
        await db.close()


async def get_next_unscored_company_result() -> dict | None:
    """Get a validated company_results row where sentiment is NULL."""
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT cr.*, a.url as article_url
            FROM company_results cr
            JOIN articles a ON cr.article_id = a.id
            WHERE cr.validated = 1 AND cr.sentiment IS NULL
            ORDER BY a.impact DESC
            LIMIT 1
        """)
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def save_company_sentiment(
    article_id: int, ticker: str, sentiment: float, impact: float
):
    """Update sentiment and impact for a company_results row."""
    db = await get_db()
    try:
        await db.execute(
            """UPDATE company_results
               SET sentiment = ?, impact = ?, classified_at = CURRENT_TIMESTAMP
               WHERE article_id = ? AND ticker = ?""",
            (sentiment, impact, article_id, ticker),
        )
        await db.commit()
    finally:
        await db.close()


async def get_company_results_for_article(article_id: int) -> list[dict]:
    """Get scored company results for an article."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM company_results WHERE article_id = ? AND sentiment IS NOT NULL ORDER BY ticker",
            (article_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# --- Settings ---


async def get_setting(key: str) -> str | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row["value"] if row else None
    finally:
        await db.close()


async def set_setting(key: str, value: str):
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        await db.commit()
    finally:
        await db.close()


# --- Fundus ---


async def add_fundus_articles(articles: list[dict]) -> int:
    """Bulk insert fundus articles with url, title, published_at, content.

    feed_id is NULL for fundus articles. Duplicates are ignored via URL unique constraint.
    Returns count of newly inserted articles.
    """
    db = await get_db()
    try:
        added = 0
        for article in articles:
            try:
                await db.execute(
                    "INSERT INTO articles (feed_id, url, title, content, published_at) VALUES (NULL, ?, ?, ?, ?)",
                    (
                        article["url"],
                        article.get("title"),
                        article.get("content"),
                        _normalize_date(article.get("published_at")),
                    ),
                )
                added += 1
            except aiosqlite.IntegrityError:
                pass
        await db.commit()
        return added
    finally:
        await db.close()


async def get_articles_needing_content(limit: int = 10) -> list[dict]:
    """Return articles that need processing but have no content yet.

    Prioritizes impact-unscored articles (biggest backlog), then unclassified,
    then unscanned.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT a.id, a.url FROM articles a
            WHERE a.content IS NULL
              AND (a.impact IS NULL
                   OR NOT EXISTS (
                       SELECT 1 FROM results r WHERE r.article_id = a.id
                   ))
            ORDER BY
                CASE WHEN a.impact IS NULL THEN 0 ELSE 1 END,
                a.fetched_at DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def store_article_content(article_id: int, content: str, published_at: str | None):
    """Store fetched content for an article."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE articles SET content = ?, published_at = COALESCE(published_at, ?) WHERE id = ?",
            (content, _normalize_date(published_at), article_id),
        )
        await db.commit()
    finally:
        await db.close()


async def clear_processed_content(model_count: int):
    """NULL out content for articles that have been fully processed.

    An article is considered processed when impact IS NOT NULL and it has
    results from at least model_count distinct models.
    """
    db = await get_db()
    try:
        await db.execute(
            """
            UPDATE articles SET content = NULL
            WHERE content IS NOT NULL
              AND impact IS NOT NULL
              AND id IN (
                  SELECT article_id FROM results
                  GROUP BY article_id
                  HAVING COUNT(DISTINCT model) >= ?
              )
        """,
            (model_count,),
        )
        await db.commit()
    finally:
        await db.close()
