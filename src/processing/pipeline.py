"""
Main detection pipeline - orchestrates all CV detectors
Simplified sync-based architecture for 1-2 Hz scanning
"""

import time
from typing import Optional
from collections import deque

import cv2
import numpy as np
from loguru import logger

from api.schemas import CVAnalysisResponse, Position, Metadata, MinimapResolution, TowerSummary
from capture.screen import ScreenCapture
from config import settings
from detection import (
    ChampionDetector,
    JungleCampDetector,
    ObjectiveDetector,
    LaneStateDetector,
    StructureDetector
)


class DetectionPipeline:
    """Coordinates all CV detection tasks - simple sequential execution"""

    def __init__(self):
        self.is_ready = False
        self.screen_capture: Optional[ScreenCapture] = None
        self.last_minimap: Optional[np.ndarray] = None
        self.minimap_region: Optional[tuple] = None

        # Screenshot storage - keep last 3 captures for debugging
        self.screenshot_buffer = deque(maxlen=3)
        self.latest_screenshot: Optional[bytes] = None

        # Initialize detectors
        self.champion_detector: Optional[ChampionDetector] = None
        self.jungle_detector: Optional[JungleCampDetector] = None
        self.objective_detector: Optional[ObjectiveDetector] = None
        self.lane_detector: Optional[LaneStateDetector] = None
        self.structure_detector: Optional[StructureDetector] = None

    async def initialize(self):
        """Initialize screen capture and all enabled detectors"""
        logger.info("Initializing detection pipeline...")

        # Initialize screen capture
        self.screen_capture = ScreenCapture()
        await self.screen_capture.initialize()

        # Determine minimap region (auto-detect or manual calibration)
        if settings.AUTO_DETECT_MINIMAP:
            # Try auto-detection first
            logger.info("🎯 Auto-detection enabled")
            self.minimap_region = await self._detect_minimap_region()

            # Fallback to manual calibration if auto-detection fails
            if self.minimap_region is None:
                logger.warning("⚠️  Auto-detection failed, falling back to manual calibration")
                logger.info("📍 Position the green window over your minimap")
                self.minimap_region = self.screen_capture.calibrate_minimap()

        elif self.screen_capture.needs_calibration():
            # Manual calibration (first run or invalid coords)
            logger.warning("⚠️  Minimap not calibrated!")
            logger.info("📍 Starting calibration - position the green window over your minimap")
            logger.info("🎮 Make sure League of Legends is running with minimap visible")

            # Run calibration (blocking) - returns calibrated coordinates
            self.minimap_region = self.screen_capture.calibrate_minimap()
            logger.success(f"✅ Calibration complete! Using region: {self.minimap_region}")

        else:
            # Use configured minimap region
            self.minimap_region = (
                settings.MINIMAP_X,
                settings.MINIMAP_Y,
                settings.MINIMAP_WIDTH,
                settings.MINIMAP_HEIGHT
            )
            logger.info(f"✅ Using calibrated minimap region: {self.minimap_region}")

        # Initialize detectors based on config
        if settings.ENABLE_CHAMPION_DETECTION:
            self.champion_detector = ChampionDetector()
            self.champion_detector.initialize()

        if settings.ENABLE_JUNGLE_DETECTION:
            self.jungle_detector = JungleCampDetector()
            self.jungle_detector.initialize()

        if settings.ENABLE_OBJECTIVE_DETECTION:
            self.objective_detector = ObjectiveDetector()
            self.objective_detector.initialize()

        if settings.ENABLE_LANE_DETECTION:
            self.lane_detector = LaneStateDetector()
            self.lane_detector.initialize()

        if settings.ENABLE_TOWER_DETECTION:
            self.structure_detector = StructureDetector()
            self.structure_detector.initialize()

        self.is_ready = True
        logger.success("Pipeline initialized and ready")

    async def cleanup(self):
        """Cleanup resources"""
        if self.screen_capture:
            await self.screen_capture.cleanup()

    async def analyze(self) -> CVAnalysisResponse:
        """
        Main analysis method - captures minimap and runs all detectors sequentially

        Returns: CVAnalysisResponse with all detected data
        """
        start_time = time.perf_counter()
        timestamp = int(time.time() * 1000)

        # Capture minimap
        minimap = await self._capture_minimap()
        if minimap is None:
            logger.warning("⚠️  Minimap capture failed - returning empty response with 0 structures")
            return self._empty_response(timestamp, 0)

        # Run detections sequentially (simple, debuggable)
        # At 2 Hz (500ms budget), we have plenty of time for all detectors

        champions = []
        jungle_camps = []
        objectives = []
        lane_states = []
        towers = []
        structures = []
        player_pos = None
        errors = []

        try:
            # Champion detection (~10-100ms depending on method)
            if self.champion_detector:
                champions = self.champion_detector.detect(minimap)

                # Extract player position if found
                for champ in champions:
                    if champ.isPlayer:
                        player_pos = champ.position
                        break

        except Exception as e:
            logger.error(f"Champion detection failed: {e}")
            errors.append(f"Champion detection error: {str(e)}")

        try:
            # Jungle camp detection (~50-150ms with template matching/OCR)
            if self.jungle_detector:
                jungle_camps = self.jungle_detector.detect(minimap)

        except Exception as e:
            logger.error(f"Jungle detection failed: {e}")
            errors.append(f"Jungle detection error: {str(e)}")

        try:
            # Objective detection (~50-100ms)
            if self.objective_detector:
                objectives = self.objective_detector.detect(minimap)

        except Exception as e:
            logger.error(f"Objective detection failed: {e}")
            errors.append(f"Objective detection error: {str(e)}")

        try:
            # Lane state detection (~50-100ms)
            if self.lane_detector:
                lane_states = self.lane_detector.detect(minimap)

        except Exception as e:
            logger.error(f"Lane detection failed: {e}")
            errors.append(f"Lane detection error: {str(e)}")

        try:
            # Structure detection (~30-50ms) - towers and inhibitors
            if self.structure_detector:
                structures = self.structure_detector.detect(minimap)
                logger.info(f"🏰 Pipeline received {len(structures)} structures from detector")
            else:
                logger.warning("⚠️  Structure detector not initialized (ENABLE_TOWER_DETECTION might be False)")

        except Exception as e:
            logger.error(f"Structure detection failed: {e}")
            errors.append(f"Structure detection error: {str(e)}")

        # Generate compact tower summary from structures
        tower_summary = self._generate_tower_summary(structures)

        # Calculate processing time
        processing_ms = (time.perf_counter() - start_time) * 1000

        # Log performance
        if processing_ms > settings.MAX_PROCESSING_TIME_MS:
            logger.warning(f"Processing took {processing_ms:.1f}ms (target: {settings.MAX_PROCESSING_TIME_MS}ms)")
        else:
            logger.debug(f"Processing completed in {processing_ms:.1f}ms")

        response = CVAnalysisResponse(
            timestamp=timestamp,
            processingTimeMs=round(processing_ms, 2),
            playerPosition=player_pos,
            champions=champions,
            jungleCamps=jungle_camps,
            objectives=objectives,
            laneStates=lane_states,
            towers=tower_summary,
            structures=structures,
            metadata=Metadata(
                minimapResolution=MinimapResolution(
                    width=minimap.shape[1],
                    height=minimap.shape[0]
                ),
                detectionErrors=errors
            )
        )

        logger.debug(f"📤 Response contains {len(structures)} structures")
        return response

    async def _capture_minimap(self) -> Optional[np.ndarray]:
        """Capture just the minimap region"""
        if not self.screen_capture or not self.minimap_region:
            return None

        x, y, w, h = self.minimap_region
        screenshot = await self.screen_capture.capture_region(x, y, w, h)

        # Convert to OpenCV format (BGR)
        if screenshot is not None:
            minimap = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.last_minimap = minimap

            # Store screenshot for debugging (resize to save memory)
            # Resize to ~200px width for preview
            height, width = minimap.shape[:2]
            scale = 200 / width
            preview_width = 200
            preview_height = int(height * scale)
            preview = cv2.resize(minimap, (preview_width, preview_height))

            # Encode as JPEG (lower quality for speed/size)
            _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.latest_screenshot = buffer.tobytes()
            self.screenshot_buffer.append(self.latest_screenshot)

            return minimap

        return self.last_minimap  # Return last known minimap if capture fails

    def _generate_tower_summary(self, structures) -> TowerSummary:
        """
        Generate compact tower count summary from detected structures

        Args:
            structures: List of Structure objects

        Returns:
            TowerSummary with counts per lane for each team
        """
        from collections import defaultdict

        order_lanes = defaultdict(int)
        chaos_lanes = defaultdict(int)

        for structure in structures:
            if not structure.isAlive:
                continue

            if structure.team == "ORDER":
                order_lanes[structure.lane] += 1
            else:  # CHAOS
                chaos_lanes[structure.lane] += 1

        return TowerSummary(
            ORDER=dict(order_lanes),
            CHAOS=dict(chaos_lanes)
        )

    async def _detect_minimap_region(self) -> Optional[tuple]:
        """
        Auto-detect minimap location on screen using edge/shape detection

        This is robust to GPU settings (saturation, contrast, gamma) because
        it uses edge detection instead of color-based detection.

        Returns:
            (x, y, width, height) tuple or None if detection fails
        """
        from capture.auto_detect import detect_minimap_region

        logger.info("🔍 Auto-detecting minimap region...")

        # Capture full screen
        screenshot = await self.screen_capture.capture_full_screen()
        if screenshot is None:
            logger.error("Failed to capture screen for auto-detection")
            return None

        # Convert PIL to OpenCV format
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Run auto-detection
        region = detect_minimap_region(screenshot_cv)

        if region is None:
            logger.warning("❌ Auto-detection failed")
            return None

        x, y, w, h = region
        logger.success(f"✅ Auto-detected minimap: ({x}, {y}) {w}x{h}")

        return region

    def _empty_response(self, timestamp: int, processing_ms: float) -> CVAnalysisResponse:
        """Return empty response when no data available"""
        return CVAnalysisResponse(
            timestamp=timestamp,
            processingTimeMs=processing_ms,
            playerPosition=None,
            champions=[],
            jungleCamps=[],
            objectives=[],
            laneStates=[],
            towers=[],
            structures=[],
            metadata=Metadata(
                minimapResolution=MinimapResolution(width=0, height=0),
                detectionErrors=["No minimap data available"]
            )
        )
