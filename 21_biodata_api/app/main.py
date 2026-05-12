from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from sqlalchemy import text

from .database import engine
from .routers.reports import router as reports_router
from .routers.samples import router as samples_router
from .routers.search import router as search_router
from .routers.sequences import router as sequences_router

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan: log startup information and dispose the engine on shutdown.

    Args:
        app: The FastAPI application instance (unused directly but required by the protocol).

    Yields:
        None — control is returned to FastAPI between startup and shutdown phases.
    """
    # Startup
    db_url_safe = str(engine.url).replace(
        str(engine.url).split("@")[0].split("//")[-1], "***", 1
    ) if "@" in str(engine.url) else str(engine.url)
    logger.info("BioData API starting up. Database: %s", db_url_safe)
    logger.info("Connection pool: pool_size=%d, max_overflow=%d", 5, 10)

    yield

    # Shutdown
    logger.info("BioData API shutting down — disposing database engine.")
    await engine.dispose()


app = FastAPI(
    title="BioData API",
    description=(
        "REST API for antibody discovery lab LIMS data. "
        "Exposes B-cell samples, antibody sequences, expression results, "
        "assay outcomes, and aggregate reports."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(samples_router)
app.include_router(sequences_router)
app.include_router(reports_router, prefix="/report", tags=["reports"])
app.include_router(search_router, prefix="/search", tags=["search"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint — verifies database connectivity with a SELECT 1 query.

    Returns:
        A JSON object with ``status`` set to ``"ok"`` and ``db`` set to
        ``"connected"`` when the database is reachable.

    Raises:
        HTTPException 503: If the database query fails (propagated as an unhandled
        exception, which FastAPI converts to a 500 by default; callers should treat
        any non-200 response as unhealthy).
    """
    from .database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))

    return {"status": "ok", "db": "connected"}
