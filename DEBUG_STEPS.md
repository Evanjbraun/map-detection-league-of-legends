# Debugging Map Boundary Detection

## What Should Happen When You Start the Service

1. **Run `run_dev.bat`**
2. **Watch for these log messages:**

```
🔍 Looking for minimap at: C:\Users\evanj\...\src\data\map\minimap.png
  - File exists: True
Analyzing minimap: 350x350 (or similar dimensions)
Mean brightness: XX.X
Found terrain boundary with XX points
✅ Detected map boundary with XX points from image
```

3. **Then open http://localhost:8765/debug**
4. **Click "Start Capture"**
5. **Look for this log when map data is requested:**

```
🗺️ Serving map data:
  - Boundary points: XX
  - First 3 points: [(x, y), (x, y), (x, y)]
  - Last 3 points: [(x, y), (x, y), (x, y)]
```

## Expected Behavior

- The boundary points should be in the range 0-100 (normalized coordinates)
- You should see dark gray lines on the debug monitor forming a diamond-ish shape
- The lines should roughly match the walkable area of the minimap

## If You See "Using fallback boundary"

This means the image detection failed. Possible reasons:
1. **File not found** - minimap.png is not at `src/data/map/minimap.png`
2. **OpenCV error** - cv2 couldn't load or process the image
3. **No contours found** - threshold is wrong

## Current Setup

- **Boundary detection:** Auto-runs when `lane_states.py` is imported
- **Black pixel threshold:** 50 (pixels darker than this = terrain)
- **Detection method:** Find all dark areas, take convex hull
- **Visualization:** Dark gray (#444444) lines on debug monitor

## Quick Test

Run this to see if boundary detection is working:
```bat
debug_boundary.bat
```

This will show you how many points were detected and what they are.
