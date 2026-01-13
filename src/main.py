"""
League CV Service - High-Performance Computer Vision Service
Provides real-time minimap analysis via WebSocket and HTTP

Priority: SPEED
- WebSocket for low-latency streaming (preferred)
- HTTP REST API for compatibility fallback
- msgpack binary serialization for minimal overhead
- Sub-50ms processing target
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from api.routes import router
from config import settings
from processing.pipeline import DetectionPipeline

# Global state
start_time = time.time()
pipeline: DetectionPipeline | None = None
active_websockets: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    global pipeline

    logger.info(f"🚀 League CV Service v{settings.VERSION}")
    logger.info(f"📡 Protocol: {settings.PROTOCOL.upper()}")
    logger.info(f"⚡ Target FPS: {settings.TARGET_FPS if settings.TARGET_FPS > 0 else 'Unlimited'}")
    logger.info(f"🎯 Max processing time: {settings.MAX_PROCESSING_TIME_MS}ms")

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
