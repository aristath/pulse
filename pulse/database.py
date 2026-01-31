import json
import aiosqlite
from pathlib import Path

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
                    (feed_id, article["url"], article.get("published_at")),
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

    Only considers articles with impact >= 0.5, ordered by highest impact first.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT a.* FROM articles a
            LEFT JOIN results r ON a.id = r.article_id AND r.model = ?
            WHERE r.id IS NULL
              AND a.impact IS NOT NULL
              AND a.impact >= 0.5
            ORDER BY a.impact DESC
            LIMIT 1
        """,
            (model,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def get_next_article_for_impact() -> dict | None:
    """Get an article that hasn't been impact-scored yet.

    Prioritizes articles that already have classification results.
    """
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT a.* FROM articles a
            WHERE a.impact IS NULL
            ORDER BY
                CASE WHEN EXISTS (
                    SELECT 1 FROM results r WHERE r.article_id = a.id
                ) THEN 0 ELSE 1 END,
                a.fetched_at DESC
            LIMIT 1
        """)
        row = await cursor.fetchone()
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
            (published_at, article_id),
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
        impact_relevant = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE impact IS NOT NULL AND impact >= 0.5"
                )
            ).fetchone()
        )[0]

        # Company scan stats
        company_scanned = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM articles WHERE scanned_aliases IS NOT NULL"
                )
            ).fetchone()
        )[0]
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

        # Build per-worker task counts keyed by processing_status keys
        per_worker = {"impact": impact_scored}
        for model_name, count in classify_counts.items():
            per_worker[f"classify:{model_name}"] = count
        per_worker["company-scanner"] = company_scanned
        # Validate/sentiment are shared across models — show on first available
        per_worker["validate:modernbert-nli"] = company_validated
        per_worker["company_sentiment:modernbert-nli"] = company_scored

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


async def get_sentiment_timeseries(group_by: str) -> list[dict]:
    """Get sentiment averaged over time, grouped by country or industry.

    Args:
        group_by: "country" groups by country key, "industry" groups by sector key.
    """
    if group_by == "country":
        label_expr = "country.key"
    elif group_by == "industry":
        label_expr = "sector.key"
    else:
        raise ValueError(f"group_by must be 'country' or 'industry', got {group_by!r}")

    sql = f"""
        SELECT CASE ((CAST(strftime('%m', a.published_at) AS INTEGER) - 1) / 3)
                   WHEN 0 THEN strftime('%Y', a.published_at) || '-01-01'
                   WHEN 1 THEN strftime('%Y', a.published_at) || '-04-01'
                   WHEN 2 THEN strftime('%Y', a.published_at) || '-07-01'
                   WHEN 3 THEN strftime('%Y', a.published_at) || '-10-01'
               END as day,
               {label_expr} as label,
               SUM(CAST(sector.value AS REAL) * COALESCE(a.impact, 0)) / MAX(SUM(a.impact), 1) as avg_sentiment
        FROM results r
        JOIN articles a ON r.article_id = a.id,
             json_each(r.signals) as country,
             json_each(country.value) as sector
        WHERE a.published_at IS NOT NULL AND r.signals != '{{}}' AND a.impact IS NOT NULL
        GROUP BY day, label
        ORDER BY day
    """
    db = await get_db()
    try:
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_sentiment_detailed() -> list[dict]:
    """Get sentiment by day, country, and industry (quarterly aggregation)."""
    sql = """
        SELECT CASE ((CAST(strftime('%m', a.published_at) AS INTEGER) - 1) / 3)
                   WHEN 0 THEN strftime('%Y', a.published_at) || '-01-01'
                   WHEN 1 THEN strftime('%Y', a.published_at) || '-04-01'
                   WHEN 2 THEN strftime('%Y', a.published_at) || '-07-01'
                   WHEN 3 THEN strftime('%Y', a.published_at) || '-10-01'
               END as day,
               country.key as country,
               sector.key as industry,
               SUM(CAST(sector.value AS REAL) * COALESCE(a.impact, 0)) / MAX(SUM(a.impact), 1) as avg_sentiment
        FROM results r
        JOIN articles a ON r.article_id = a.id,
             json_each(r.signals) as country,
             json_each(country.value) as sector
        WHERE a.published_at IS NOT NULL AND r.signals != '{}' AND a.impact IS NOT NULL
        GROUP BY day, country, industry
        ORDER BY day
    """
    db = await get_db()
    try:
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_company_sentiment_timeseries() -> list[dict]:
    """Get company sentiment over time, grouped by ticker (quarterly)."""
    sql = """
        SELECT CASE ((CAST(strftime('%m', a.published_at) AS INTEGER) - 1) / 3)
                   WHEN 0 THEN strftime('%Y', a.published_at) || '-01-01'
                   WHEN 1 THEN strftime('%Y', a.published_at) || '-04-01'
                   WHEN 2 THEN strftime('%Y', a.published_at) || '-07-01'
                   WHEN 3 THEN strftime('%Y', a.published_at) || '-10-01'
               END as day,
               cr.ticker as label,
               AVG(cr.sentiment) as avg_sentiment
        FROM company_results cr
        JOIN articles a ON cr.article_id = a.id
        WHERE a.published_at IS NOT NULL
          AND cr.sentiment IS NOT NULL
        GROUP BY day, cr.ticker
        ORDER BY day
    """
    db = await get_db()
    try:
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
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
                        article.get("published_at"),
                    ),
                )
                added += 1
            except aiosqlite.IntegrityError:
                pass
        await db.commit()
        return added
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
