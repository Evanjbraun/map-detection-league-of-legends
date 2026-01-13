"""Configuration with pydantic-settings"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Service
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8765
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Protocol
    PROTOCOL: Literal["websocket", "http"] = "websocket"
    ENABLE_HTTP_FALLBACK: bool = True

    # Detection
    ENABLE_CHAMPION_DETECTION: bool = True
    ENABLE_JUNGLE_DETECTION: bool = True
    ENABLE_OBJECTIVE_DETECTION: bool = True
    ENABLE_LANE_DETECTION: bool = True
    ENABLE_TOWER_DETECTION: bool = True

    # Performance
    PROCESSING_THREADS: int = 4
    TARGET_FPS: int = 30  # 0 = unlimited
    MAX_PROCESSING_TIME_MS: int = 50
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.6

    # Screen Capture
    CAPTURE_METHOD: Literal["mss", "win32", "pillow"] = "mss"
    AUTO_DETECT_MINIMAP: bool = True
    MINIMAP_CORNER: Literal["bottom_right", "bottom_left"] = "bottom_right"

    # Calibration
    MINIMAP_X: int = 0
    MINIMAP_Y: int = 0
    MINIMAP_WIDTH: int = 250
    MINIMAP_HEIGHT: int = 250

    # Template Matching
    TEMPLATE_MATCH_THRESHOLD: float = 0.7
    USE_GPU_ACCELERATION: bool = False

    # OCR
    OCR_ENGINE: Literal["tesseract", "easyocr"] = "easyocr"
    TESSERACT_PATH: str = ""

    # Serialization
    SERIALIZATION_FORMAT: Literal["json", "msgpack"] = "msgpack"


settings = Settings()

# Debug: Log configuration on import
if __name__ != "__main__":
    from loguru import logger
    logger.debug(f"Config loaded from: {ENV_FILE}")
    logger.debug(f".env exists: {ENV_FILE.exists()}")
    logger.debug(f"ENABLE_TOWER_DETECTION: {settings.ENABLE_TOWER_DETECTION}")
