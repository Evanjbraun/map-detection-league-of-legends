"""
League CV Service - High-Performance Computer Vision Service
Provides real-time minimap analysis via WebSocket and HTTP

Priority: SPEED
- WebSocket for low-latency streaming (preferred)
- HTTP REST API for compatibility fallback
- msgpack binary serialization for minimal overhead
- Sub-50ms processing target
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Suppress uvicorn's verbose WebSocket DEBUG logs (< TEXT '...' [N bytes])
logging.getLogger("uvicorn.protocols.websockets.websockets_impl").setLevel(logging.WARNING)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from api.routes import router
from config import settings, PROJECT_ROOT
from processing.pipeline import DetectionPipeline

# Global state
start_time = time.time()
pipeline: DetectionPipeline | None = None
active_websockets: set[WebSocket] = set()

# Debug logging state
debug_log_file = None
last_debug_log_time = 0
DEBUG_LOG_INTERVAL = 5  # Log every 5 seconds


def cleanup_old_logs(logs_dir: Path, max_files: int = 3) -> None:
    """Remove oldest log files if more than max_files exist"""
    log_files = sorted(logs_dir.glob("cv_output_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)

    if len(log_files) > max_files:
        for old_file in log_files[max_files:]:
            logger.info(f"🗑️ Removing old log file: {old_file.name}")
            old_file.unlink()


def init_debug_log() -> Path | None:
    """Initialize debug log file for this session"""
    if not settings.DEBUG:
        return None

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Cleanup old logs (keep max 3)
    cleanup_old_logs(logs_dir, max_files=3)

    # Create new log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"cv_output_{timestamp}.jsonl"

    logger.info(f"📝 Debug log file: {log_file}")
    return log_file


def log_debug_output(result_dict: dict) -> None:
    """Append result to debug log file (every 5 seconds)"""
    global last_debug_log_time, debug_log_file

    if debug_log_file is None:
        return

    current_time = time.time()
    if current_time - last_debug_log_time < DEBUG_LOG_INTERVAL:
        return

    last_debug_log_time = current_time

    try:
        with open(debug_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write debug log: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    global pipeline, debug_log_file

    logger.info(f"🚀 League CV Service v{settings.VERSION}")
    logger.info(f"📡 Protocol: {settings.PROTOCOL.upper()}")
    logger.info(f"⚡ Target FPS: {settings.TARGET_FPS if settings.TARGET_FPS > 0 else 'Unlimited'}")
    logger.info(f"🎯 Max processing time: {settings.MAX_PROCESSING_TIME_MS}ms")

    # Initialize debug logging (only in DEBUG mode)
    debug_log_file = init_debug_log()

    # Initialize detection pipeline
    try:
        pipeline = DetectionPipeline()
        await pipeline.initialize()
        logger.success("✅ Detection pipeline ready")
    except Exception as e:
        logger.error(f"❌ Pipeline init failed: {e}")
        raise

    yield

    # Cleanup
    logger.info("🛑 Shutting down")
    if pipeline:
        await pipeline.cleanup()


app = FastAPI(
    title="League CV Service",
    description="Ultra-fast local CV for League minimap analysis",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS for Electron app and debug page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debug page
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include HTTP routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "League CV Service",
        "version": settings.VERSION,
        "protocol": settings.PROTOCOL,
        "uptime": round(time.time() - start_time, 2),
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "websocket": "WS /ws",
            "debug": "GET /debug"
        }
    }


@app.get("/debug")
async def debug_page():
    """Serve the debug visualization page"""
    debug_html = Path(__file__).parent.parent / "debug.html"
    if debug_html.exists():
        return FileResponse(debug_html)
    return {"error": "Debug page not found"}


@app.get("/health")
async def health():
    """Health check for TypeScript client"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "uptime_seconds": round(time.time() - start_time, 2),
        "pipeline_ready": pipeline is not None and pipeline.is_ready,
        "active_connections": len(active_websockets),
        "protocol": settings.PROTOCOL
    }


@app.get("/templates/champions/{champion_name}")
async def get_champion_icon(champion_name: str):
    """
    Serve champion template images for debug monitor

    Normalizes champion name (lowercase, no spaces/apostrophes) to match template filenames
    Tries multiple file extensions (.jpg, .png, .webp)
    """
    templates_dir = PROJECT_ROOT / "models" / "templates" / "champions"

    # Normalize champion name to match template filenames
    # "Kai'Sa" -> "kaisa", "Lee Sin" -> "leesin", "Brand" -> "brand"
    normalized_name = champion_name.replace("'", "").replace(" ", "").lower()

    # Try different extensions
    for ext in [".jpg", ".png", ".webp"]:
        icon_path = templates_dir / f"{normalized_name}{ext}"
        if icon_path.exists():
            return FileResponse(
                icon_path,
                media_type=f"image/{ext[1:]}",  # Remove the dot
                headers={"Cache-Control": "public, max-age=86400"}  # Cache for 24 hours
            )

    # Return 404 if not found
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Champion icon not found: {champion_name} (normalized: {normalized_name})")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming (FASTEST)

    Client sends: {} (empty object triggers analysis)
    Server sends: CVAnalysisResponse (msgpack or json)

    This is the preferred communication method for speed.
    """
    await websocket.accept()
    active_websockets.add(websocket)

    logger.info(f"🔌 WebSocket connected (total: {len(active_websockets)})")

    try:
        while True:
            # Wait for client trigger (empty message = "analyze now")
            data = await websocket.receive_text()

            if not pipeline or not pipeline.is_ready:
                await websocket.send_json({
                    "error": "Pipeline not ready",
                    "timestamp": int(time.time() * 1000)
                })
                continue

            # Run analysis
            start = time.perf_counter()
            result = await pipeline.analyze()
            processing_time = (time.perf_counter() - start) * 1000

            # Log what we're sending
            result_dict = result.model_dump()
            logger.debug(f"🔌 WebSocket sending: {len(result_dict.get('structures', []))} structures, "
                        f"{len(result_dict.get('champions', []))} champions, "
                        f"{len(result_dict.get('laneStates', []))} lane states")

            # Write to debug log file (every 5 seconds in DEBUG mode)
            log_debug_output(result_dict)

            # Send result (msgpack if enabled, else json)
            if settings.SERIALIZATION_FORMAT == "msgpack":
                import msgpack
                await websocket.send_bytes(msgpack.packb(result_dict))
            else:
                await websocket.send_json(result_dict)

            # Log if slow
            if processing_time > settings.MAX_PROCESSING_TIME_MS:
                logger.warning(f"⚠️ Slow frame: {processing_time:.1f}ms")

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected (total: {len(active_websockets) - 1})")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)


if __name__ == "__main__":
    logger.info("🔥 Starting in development mode")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
        # Performance tuning
        ws_ping_interval=None,  # Disable WS pings for speed
        ws_ping_timeout=None,
        timeout_keep_alive=300,
    )
