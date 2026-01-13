"""
Test utility for minimap auto-detection

Usage:
    python test_autodetect.py <screenshot_path>

This will:
1. Load the screenshot
2. Run auto-detection
3. Display the detected region coordinates
4. Save an annotated image showing the detection
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from capture.auto_detect import test_detection_on_screenshot
from loguru import logger


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_autodetect.py <screenshot_path>")
        print()
        print("Example:")
        print("  python test_autodetect.py minimap_screenshot.png")
        sys.exit(1)

    screenshot_path = sys.argv[1]

    if not Path(screenshot_path).exists():
        logger.error(f"Screenshot not found: {screenshot_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MINIMAP AUTO-DETECTION TEST")
    logger.info("=" * 60)

    # Run detection
    region = test_detection_on_screenshot(screenshot_path, debug_output=True)

    if region:
        x, y, w, h = region
        logger.success("=" * 60)
        logger.success("DETECTION SUCCESSFUL!")
        logger.success("=" * 60)
        logger.success(f"Position: ({x}, {y})")
        logger.success(f"Size: {w}x{h}")
        logger.success("")
        logger.success("To use these coordinates in your .env:")
        logger.success(f"MINIMAP_X={x}")
        logger.success(f"MINIMAP_Y={y}")
        logger.success(f"MINIMAP_WIDTH={w}")
        logger.success(f"MINIMAP_HEIGHT={h}")
        logger.success("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("DETECTION FAILED")
        logger.error("=" * 60)
        logger.error("Could not automatically detect minimap region.")
        logger.error("Please use manual calibration instead.")
        logger.error("=" * 60)


if __name__ == "__main__":
    main()
