"""HTTP API routes (fallback to WebSocket)"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from loguru import logger

from api.schemas import CVAnalysisResponse
from processing.pipeline import DetectionPipeline

router = APIRouter()

# This will be injected from main.py
pipeline: DetectionPipeline | None = None


def get_pipeline() -> DetectionPipeline:
    """Dependency to access pipeline"""
    from main import pipeline as global_pipeline
    if global_pipeline is None or not global_pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return global_pipeline


@router.post("/analyze", response_model=CVAnalysisResponse)
async def analyze_minimap(pipeline: DetectionPipeline = Depends(get_pipeline)):
    """
    HTTP endpoint for minimap analysis (fallback if WebSocket not available)

    NOTE: WebSocket is preferred for performance (lower latency, less overhead)
    """
    try:
        result = await pipeline.analyze()
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibration")
async def get_calibration(pipeline: DetectionPipeline = Depends(get_pipeline)):
    """Get current minimap calibration"""
    return {
        "minimapRegion": pipeline.minimap_region,
        "autoDetected": pipeline.screen_capture is not None
    }


@router.post("/calibrate")
async def trigger_calibration(pipeline: DetectionPipeline = Depends(get_pipeline)):
    """
    Trigger minimap recalibration

    Opens the calibration UI and updates minimap region
    """
    try:
        logger.info("Recalibration requested via API")

        # Run calibration UI (blocking on main thread)
        new_region = pipeline.screen_capture.calibrate_minimap()

        # Update pipeline's minimap region
        pipeline.minimap_region = new_region

        logger.success(f"Recalibration complete: {new_region}")

        return {
            "success": True,
            "minimapRegion": new_region,
            "message": "Calibration complete"
        }

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@router.get("/screenshot")
async def get_screenshot(pipeline: DetectionPipeline = Depends(get_pipeline)):
    """
    Get the latest minimap screenshot preview

    Returns a downsized JPEG preview of what the service is actually capturing
    """
    if pipeline.latest_screenshot is None:
        raise HTTPException(status_code=404, detail="No screenshot available yet")

    return Response(
        content=pipeline.latest_screenshot,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


