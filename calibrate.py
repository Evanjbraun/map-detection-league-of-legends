"""
Run minimap calibration tool

This will show a draggable green overlay that you position over the minimap
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from capture.calibration import calibrate_minimap_region
from config import settings

if __name__ == "__main__":
    print("\n🎯 Minimap Calibration Tool")
    print("\nMake sure League of Legends is running and visible!")
    print("Press Enter to start...")
    input()

    # Get current settings
    current_x = settings.MINIMAP_X
    current_y = settings.MINIMAP_Y
    current_width = settings.MINIMAP_WIDTH
    current_height = settings.MINIMAP_HEIGHT

    print(f"\nCurrent settings: ({current_x}, {current_y}) {current_width}x{current_height}")

    # Run calibration
    x, y, width, height = calibrate_minimap_region(current_x, current_y, current_width, current_height)

    print("\n" + "="*60)
    print("✅ CALIBRATION COMPLETE!")
    print("="*60)
    print(f"\nNew coordinates: ({x}, {y}) {width}x{height}")
    print("\nUpdate your src/.env file with:")
    print(f"\nMINIMAP_X={x}")
    print(f"MINIMAP_Y={y}")
    print(f"MINIMAP_WIDTH={width}")
    print(f"MINIMAP_HEIGHT={height}")
    print("\n" + "="*60)
