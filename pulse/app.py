import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import psutil

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel as PydanticModel
from sse_starlette.sse import EventSourceResponse

from pulse import database as db
from pulse.classifier import ensemble, processing_status, get_worker_avg_times, start_workers, stop_workers, WORKER_LABELS
from pulse.fetcher import fetch_all_feeds
from pulse.fundus_crawler import (
    get_available_publishers,
    get_enabled_publisher_ids,
    crawl_enabled_publishers,
)
from pulse.scheduler import start_scheduler, stop_scheduler


def _model_count() -> int:
    """Count models that do classify work (write to results table)."""
    from pulse.classifier import WORKER_CONFIGS
    return max(sum(1 for _, caps in WORKER_CONFIGS if any(t == "classify" for t, _ in caps)), 1)


DISPLAY_ORDER = [
    "impact",
    "classify:deberta",
    "company_sentiment:deberta-aux",
    "company-scanner",
    "validate:deberta-aux",
]


def _merge_impact_workers(processing: dict, avg_times: dict) -> tuple[dict, dict]:
    """Collapse impact-1/2/3 into a single 'impact' entry for the UI."""
    merged_proc = {}
    merged_avg = {}
    impact_entries = []
    impact_avgs = []
    for key, val in processing.items():
        if key.startswith("impact:impact-"):
            impact_entries.append(val)
            if avg_times.get(key) is not None:
                impact_avgs.append(avg_times[key])
        else:
            merged_proc[key] = val
            merged_avg[key] = avg_times.get(key)
    # Pick the most recently started active impact worker, or idle
    active = [e for e in impact_entries if e.get("article_id")]
    if active:
        best = max(active, key=lambda e: e.get("started_at") or 0)
    else:
        best = {"article_id": None, "article_url": None, "started_at": None}
    merged_proc["impact"] = best
    merged_avg["impact"] = round(sum(impact_avgs) / len(impact_avgs), 1) if impact_avgs else None
    # Reorder to match pipeline flow
    ordered_proc = {k: merged_proc[k] for k in DISPLAY_ORDER if k in merged_proc}
    ordered_avg = {k: merged_avg[k] for k in DISPLAY_ORDER if k in merged_avg}
    return ordered_proc, ordered_avg


logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _load_models_background():
    try:
        ensemble.load_models()
    except Exception:
        logger.exception("Failed to load models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    logger.info("Database initialized")
    thread = threading.Thread(target=_load_models_background, daemon=True)
    thread.start()
    logger.info("Model loading started in background")
    start_scheduler()
    start_workers()
    yield
    await stop_workers()
    stop_scheduler()
    logger.info("Shutdown complete")


# --- Pydantic models ---


class FeedCreate(PydanticModel):
    url: str
    name: str | None = None


class FeedUpdate(PydanticModel):
    enabled: bool | None = None
    name: str | None = None


# --- App ---

app = FastAPI(title="Pulse", docs_url="/docs", lifespan=lifespan)


# --- HTML Pages ---


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats = await db.get_stats(_model_count())
    processing, avg_times = _merge_impact_workers(processing_status, get_worker_avg_times())
    feeds = await db.get_feeds()
    sentinel_url = await db.get_setting("sentinel_url") or ""
    prompt_impact = await db.get_setting("prompt_impact") or ""
    prompt_country = await db.get_setting("prompt_country") or ""
    prompt_sentiment = await db.get_setting("prompt_sentiment") or ""
    prompt_sector = await db.get_setting("prompt_sector") or ""
    prompt_company = await db.get_setting("prompt_company") or ""
    classify_threshold = await db.get_setting("classify_threshold") or "0.5"
    scan_threshold = await db.get_setting("scan_threshold") or "0.3"
    validate_threshold = await db.get_setting("validate_threshold") or "0.5"
    top_articles = await db.get_top_impactful_articles()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "processing": processing,
            "avg_times": avg_times,
            "worker_labels": WORKER_LABELS,
            "feeds": feeds,
            "sentinel_url": sentinel_url,
            "prompt_impact": prompt_impact,
            "prompt_country": prompt_country,
            "prompt_sentiment": prompt_sentiment,
            "prompt_sector": prompt_sector,
            "prompt_company": prompt_company,
            "classify_threshold": classify_threshold,
            "scan_threshold": scan_threshold,
            "validate_threshold": validate_threshold,
            "top_articles": top_articles,
        },
    )





# --- API Endpoints ---


@app.get("/api/feeds")
async def api_list_feeds():
    return await db.get_feeds()


@app.post("/api/feeds")
async def api_add_feed(
    request: Request,
    url: str = Form(None),
    name: str = Form(None),
):
    # Support both form data (HTMX) and JSON (API)
    if url is None:
        body = await request.json()
        url = body["url"]
        name = body.get("name")
    feed_id = await db.add_feed(url, name or None)
    return {"id": feed_id}


@app.delete("/api/feeds/{feed_id}")
async def api_delete_feed(feed_id: int):
    await db.delete_feed(feed_id)
    return {"ok": True}


@app.patch("/api/feeds/{feed_id}")
async def api_update_feed(feed_id: int, update: FeedUpdate):
    kwargs = {k: v for k, v in update.model_dump().items() if v is not None}
    if kwargs:
        await db.update_feed(feed_id, **kwargs)
    return {"ok": True}


@app.post("/api/articles")
async def api_add_article(
    request: Request,
    url: str = Form(None),
):
    if url is None:
        body = await request.json()
        url = body["url"]
    article_id = await db.add_article_url(url.strip())
    if article_id is None:
        return {"ok": False, "error": "duplicate"}
    return {"ok": True, "id": article_id}


@app.get("/api/articles")
async def api_list_articles(limit: int = 50, offset: int = 0, processed: bool = False):
    return await db.get_articles(limit=limit, offset=offset, processed_only=processed)


@app.get("/api/articles/{article_id}/results")
async def api_article_results(article_id: int):
    return await db.get_results_for_article(article_id)


@app.get("/api/articles/{article_id}/company-results")
async def api_article_company_results(article_id: int):
    return await db.get_company_results_for_article(article_id)


@app.get("/api/charts/sentiment-bars")
async def api_charts_sentiment_bars(type: str = "country", days: int = 90, decay: bool = True):
    if type not in ("country", "industry", "company", "detailed"):
        raise HTTPException(400, "type must be 'country', 'industry', 'company', or 'detailed'")
    days = max(1, min(days, 3650))
    threshold = float(await db.get_setting("classify_threshold") or "0.5")
    return await db.get_sentiment_bars(type, threshold, days=days, decay=decay)


@app.get("/api/charts/sentiment-bar-articles")
async def api_charts_sentiment_bar_articles(type: str, label: str, days: int = 90, decay: bool = True):
    if type not in ("country", "industry", "company"):
        raise HTTPException(400, "type must be 'country', 'industry', or 'company'")
    days = max(1, min(days, 3650))
    threshold = float(await db.get_setting("classify_threshold") or "0.5")
    return await db.get_sentiment_bar_articles(type, label, threshold, days=days, decay=decay)


@app.get("/api/charts/articles-by-month")
async def api_charts_articles_by_month():
    return await db.get_articles_by_month()


@app.get("/api/stats")
async def api_stats():
    return await db.get_stats(_model_count())


@app.get("/api/stats/processing")
async def api_processing():
    return {"processing": processing_status, "avg_times": get_worker_avg_times()}


_cached_stats: dict = {"cpu_percent": 0.0, "ram_percent": 0.0, "temperature": None}


def _get_cpu_temp() -> float | None:
    """Get CPU package/die temperature, preferring the most relevant sensor."""
    temps = getattr(psutil, "sensors_temperatures", lambda: None)()
    if not temps:
        return None
    # Prefer coretemp (Intel) or k10temp (AMD) package sensor
    for chip in ("coretemp", "k10temp"):
        if chip in temps:
            for entry in temps[chip]:
                if "package" in (entry.label or "").lower():
                    return entry.current
            # No package label — take first entry
            if temps[chip]:
                return temps[chip][0].current
    # Fallback: x86_pkg_temp zone
    if "x86_pkg_temp" in temps and temps["x86_pkg_temp"]:
        return temps["x86_pkg_temp"][0].current
    return None


def _sample_stats():
    """Background sampler — runs every 2s in a daemon thread."""
    # Prime the non-blocking CPU counter
    psutil.cpu_percent(interval=None)
    import time

    while True:
        time.sleep(2)
        _cached_stats["cpu_percent"] = psutil.cpu_percent(interval=None)
        _cached_stats["ram_percent"] = psutil.virtual_memory().percent
        _cached_stats["temperature"] = _get_cpu_temp()


threading.Thread(target=_sample_stats, daemon=True).start()


def _get_system_stats() -> dict:
    return dict(_cached_stats)


@app.get("/api/system-stats")
async def api_system_stats():
    return _get_system_stats()


@app.post("/api/fetch")
async def api_trigger_fetch():
    """Manually trigger RSS feed fetch."""
    await fetch_all_feeds()
    return {"ok": True}


# --- Fundus ---


@app.get("/api/fundus/publishers")
async def api_fundus_publishers():
    """Return full publisher catalog with enabled flags."""
    catalog = get_available_publishers()
    enabled = set(await get_enabled_publisher_ids())
    result = {}
    for country, publishers in catalog.items():
        result[country] = [{**p, "enabled": p["id"] in enabled} for p in publishers]
    return result


@app.put("/api/fundus/publishers")
async def api_set_fundus_publishers(request: Request):
    """Save enabled fundus publisher IDs."""
    body = await request.json()
    publisher_ids = body.get("publishers", [])
    await db.set_setting("fundus_publishers", json.dumps(publisher_ids))
    return {"ok": True}


@app.post("/api/fundus/crawl")
async def api_trigger_fundus_crawl():
    """Manually trigger fundus crawl."""
    added = await crawl_enabled_publishers()
    return {"ok": True, "added": added}


@app.get("/api/settings/sentinel_url")
async def api_get_sentinel_url():
    val = await db.get_setting("sentinel_url")
    return {"sentinel_url": val or ""}


@app.put("/api/settings/sentinel_url")
async def api_set_sentinel_url(
    request: Request,
    url: str = Form(None),
):
    if url is None:
        body = await request.json()
        url = body.get("url", "")
    url = url.strip().rstrip("/")
    await db.set_setting("sentinel_url", url)
    # Refresh labels immediately
    await ensemble.fetch_labels()
    return {"ok": True}


@app.put("/api/settings/prompts")
async def api_set_prompts(
    request: Request,
    prompt_impact: str = Form(None),
    prompt_country: str = Form(None),
    prompt_sector: str = Form(None),
    prompt_sentiment: str = Form(None),
    prompt_company: str = Form(None),
):
    if prompt_impact is None:
        body = await request.json()
        prompt_impact = body.get("prompt_impact", "")
        prompt_country = body.get("prompt_country", "")
        prompt_sector = body.get("prompt_sector", "")
        prompt_sentiment = body.get("prompt_sentiment", "")
        prompt_company = body.get("prompt_company", "")
    await db.set_setting("prompt_impact", (prompt_impact or "").strip())
    await db.set_setting("prompt_country", (prompt_country or "").strip())
    await db.set_setting("prompt_sector", (prompt_sector or "").strip())
    await db.set_setting("prompt_sentiment", (prompt_sentiment or "").strip())
    await db.set_setting("prompt_company", (prompt_company or "").strip())
    return {"ok": True}


@app.put("/api/settings/thresholds")
async def api_set_thresholds(
    request: Request,
    classify_threshold: str = Form(None),
    scan_threshold: str = Form(None),
    validate_threshold: str = Form(None),
):
    if scan_threshold is None:
        body = await request.json()
        classify_threshold = body.get("classify_threshold", "0.5")
        scan_threshold = body.get("scan_threshold", "0.3")
        validate_threshold = body.get("validate_threshold", "0.5")
    await db.set_setting("classify_threshold", (classify_threshold or "0.5").strip())
    await db.set_setting("scan_threshold", (scan_threshold or "0.3").strip())
    await db.set_setting("validate_threshold", (validate_threshold or "0.5").strip())
    return {"ok": True}


# --- SSE ---


@app.get("/api/events")
async def sse_events(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            stats = await db.get_stats(_model_count())
            processing, avg_times = _merge_impact_workers(processing_status, get_worker_avg_times())
            data = {
                "stats": stats,
                "processing": processing,
                "avg_times": avg_times,
                "worker_labels": WORKER_LABELS,
            }
            yield {"event": "update", "data": json.dumps(data)}
            await asyncio.sleep(3)

    return EventSourceResponse(event_generator())


# --- HTMX Partials ---


@app.get("/partials/system-stats", response_class=HTMLResponse)
async def partial_system_stats(request: Request):
    stats = _get_system_stats()
    return templates.TemplateResponse(
        "partials/system_stats.html",
        {
            "request": request,
            "sys": stats,
        },
    )


@app.get("/partials/feeds", response_class=HTMLResponse)
async def partial_feeds(request: Request):
    feeds = await db.get_feeds()
    return templates.TemplateResponse(
        "partials/feeds_list.html",
        {
            "request": request,
            "feeds": feeds,
        },
    )
