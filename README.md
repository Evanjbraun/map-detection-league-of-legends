# League CV Service

Real-time computer vision service for League of Legends minimap analysis. Captures and analyzes the in-game minimap at 30 FPS to detect champion positions, jungle camps, objectives, towers, and lane states.

**Tech Stack:** FastAPI, WebSockets, OpenCV, NumPy, MSS (screen capture), Pydantic

## Quick Start

```bash
# Install
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Run (normal mode)
.\run_dev.bat

# Run with debug overlay (recommended for development)
.\run_dev_debug.bat
```

Expected output:
```
🚀 League CV Service v1.0.0
📡 Protocol: WEBSOCKET
✅ Detection pipeline ready
```

## Minimap Detection

The service automatically detects your League minimap using **edge/shape-based computer vision** - no manual setup required!

### How Auto-Detection Works

**Automatic detection** is enabled by default (`AUTO_DETECT_MINIMAP=true`):
- Uses Canny edge detection to find the minimap border
- Looks for rectangular shape in bottom-right quadrant
- Validates size (200-600px range) and color variation
- **Robust to GPU settings** (saturation, contrast, gamma) because it detects structure, not colors

**On first run:**
1. Start a League game (Practice Tool recommended)
2. Run `.\run_dev.bat`
3. Service automatically detects minimap region
4. If detection fails, falls back to manual calibration

### Manual Calibration Fallback

If auto-detection fails, a semi-transparent **GREEN window** appears:
1. **Drag it over your minimap** (drag from center)
2. **Resize to fit** (drag from edges/corners)
3. Click **LOCK** → **SAVE**
4. Service continues running automatically!

**Need to recalibrate?** Click the **"Recalibrate Minimap"** button in the debug monitor anytime (http://localhost:8765/debug).

### Testing Auto-Detection

Test the auto-detection on a screenshot:
```bash
cd src
python test_autodetect.py path/to/screenshot.png
```

This will show detected coordinates and save an annotated image.

## Debug Tools

### 1. Debug Overlay Window (New!)

**When to use:** Active development, testing detection algorithms in-game

**Start with:**
```bash
.\run_dev_debug.bat
# or
python src/main.py --debug
```

**Features:**
- Semi-transparent draggable window
- Shows live JSON responses (last 5 scans)
- Scan stats: processing time, errors, timestamp
- Play/Pause button to freeze output
- Auto-scrolling text display

**Perfect for:** Seeing raw detection data while playing League on a second monitor.

### 2. Debug Web Page

**When to use:** Visual debugging, tuning detection parameters

**Access:** http://localhost:8765/debug

**Features:**
- Live minimap capture with detection overlays
- Performance metrics (FPS, processing time, latency)
- Color-coded markers: 🟢 Player, 🔵 Allies, 🔴 Enemies, 🟡 Jungle camps, 🟣 Objectives, 🔵 Towers
- Detection counts and recent detections list
- **Raw JSON viewer** with syntax highlighting (live-updated with each frame)
- Copy/Clear controls for easy JSON inspection
- Console log for connection status and events

**Usage:**
1. Start the service: `.\run_dev_debug.bat`
2. Open http://localhost:8765/debug on your second monitor
3. Start League (Practice Tool recommended)
4. Click "Start Capture" - you'll see 30 FPS live updates with JSON payloads in the right panel

## API Usage

### WebSocket (Recommended)

```javascript
const ws = new WebSocket('ws://localhost:8765/ws');

ws.onopen = () => ws.send('{}');  // Request analysis

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Champions:', data.champions);
    console.log('Processing:', data.processingTimeMs + 'ms');
};

// Request at 30 FPS
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('{}');
}, 1000 / 30);
```

### HTTP REST

```bash
curl -X POST http://localhost:8765/analyze
```

### Response Format

```javascript
{
    "timestamp": 1234567890,
    "processingTimeMs": 12.3,
    "playerPosition": { "x": 128, "y": 234 },
    "champions": [
        { "championId": "unknown", "position": { "x": 145, "y": 220 }, "team": "blue", "confidence": 0.8 }
    ],
    "jungleCamps": [
        { "campType": "blue_buff", "position": { "x": 67, "y": 89 }, "status": "alive", "confidence": 0.9 }
    ],
    "objectives": [...],
    "laneStates": [...],
    "towers": [...],
    "metadata": { "minimapResolution": { "width": 250, "height": 250 }, "detectionErrors": [] }
}
```

## Configuration

Copy `.env.example` to `.env`:

```env
# Server
HOST=127.0.0.1
PORT=8765

# Performance
TARGET_FPS=30                    # 0 = unlimited
MAX_PROCESSING_TIME_MS=50

# Feature toggles
ENABLE_CHAMPION_DETECTION=true
ENABLE_JUNGLE_DETECTION=true
ENABLE_OBJECTIVE_DETECTION=true
ENABLE_LANE_DETECTION=true
ENABLE_TOWER_DETECTION=true

# Screen capture
AUTO_DETECT_MINIMAP=true
MINIMAP_CORNER=bottom_right

# Manual coordinates (if auto-detect fails)
MINIMAP_X=1600
MINIMAP_Y=800
MINIMAP_WIDTH=250
MINIMAP_HEIGHT=250
```

## Development Workflow

**Real-time iteration:**
1. Start service: `.\run_dev.bat`
2. Open debug page: http://localhost:8765/debug (2nd monitor)
3. Start League Practice Tool
4. Edit code in `src/processing/pipeline.py`
5. Service auto-reloads → see changes instantly on debug canvas

**Example - Implement champion detection:**

```python
# In src/processing/pipeline.py (line ~172)

async def _detect_champions(self, minimap: np.ndarray):
    """Detect champion positions using color + blob detection"""
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

    # Blue team detection
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    champions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                champions.append(ChampionSighting(
                    championId="unknown",
                    position=Position(x=cx, y=cy),
                    team="blue",
                    confidence=0.8
                ))

    return champions
```

Save and watch blue dots appear on the debug canvas!

## Project Structure

```
league-cv-service/
├── src/
│   ├── main.py              # FastAPI server + WebSocket/HTTP endpoints
│   ├── config.py            # Settings (Pydantic)
│   ├── api/
│   │   ├── routes.py        # HTTP endpoints
│   │   └── schemas.py       # Response models
│   ├── capture/
│   │   └── screen.py        # MSS screen capture (100+ FPS)
│   ├── processing/
│   │   └── pipeline.py      # ⭐ Main detection pipeline - IMPLEMENT HERE
│   └── models/
│       ├── templates/       # Template images
│       └── weights/         # ML model weights
├── debug.html               # Real-time debug visualization
├── requirements.txt
├── .env.example
└── run_dev.bat
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/debug` | GET | Debug visualization page |
| `/analyze` | POST | HTTP analysis |
| `/ws` | WebSocket | Real-time streaming |
| `/docs` | GET | Swagger docs |

## Performance Targets

| Metric | Target | Good | Bad |
|--------|--------|------|-----|
| FPS | 30 | >25 | <15 |
| Processing Time | <50ms | <50ms | >100ms |
| Screen Capture | <10ms | <10ms | >20ms |

**Optimizations:** MSS capture (~2ms/frame), parallel detection (asyncio), WebSocket, msgpack serialization, configurable features, timeout protection

## Current Status

**✅ Complete:**
- FastAPI server (WebSocket + HTTP)
- Screen capture (MSS)
- Detection pipeline orchestrator
- API schemas
- Debug visualization tool
- Config system
- **Minimap auto-detection** (edge/shape-based CV)

**⏳ TODO:**
- Implement detection algorithms in `src/processing/pipeline.py`:
  - Champion detection (color + blob)
  - Jungle camps (template matching + color)
  - Objectives (Dragon/Baron/Herald)
  - Lane states (minion waves)
  - Towers (template matching)
  - OCR for timers

## Troubleshooting

**Port in use:** `netstat -ano | findstr :8765` → `taskkill /PID <pid> /F`

**Service won't start:** `python test_startup.py`

**No detections:** Normal! Implement algorithms in `src/processing/pipeline.py`

**Can't find minimap:** Set manual coords in `.env`:
```env
AUTO_DETECT_MINIMAP=false
MINIMAP_X=1600
MINIMAP_Y=800
MINIMAP_WIDTH=250
MINIMAP_HEIGHT=250
```

**Low FPS:** Disable unused detections in `.env` or reduce `TARGET_FPS`

## Tips

1. Use Practice Tool (no time pressure)
2. Start with champion detection (easiest)
3. Tune HSV ranges for your graphics settings
4. Keep processing under 50ms
5. Enable only needed detections during dev
6. Use browser console (F12) for debugging
