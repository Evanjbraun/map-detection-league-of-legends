# League CV Service

Real-time computer vision service for League of Legends minimap analysis. Captures and analyzes the in-game minimap to detect champion positions, jungle camps, objectives, and structures.

**Tech Stack:** Python, FastAPI, WebSockets, OpenCV, NumPy, MSS (screen capture), Pydantic

## Features

- **Champion Detection** - Identifies all 10 champions in-game using Riot's local API + template matching
- **Jungle Camp Tracking** - Detects all jungle camps with position-based classification
- **Objective Detection** - Dragon, Baron, Herald status
- **Structure Detection** - Tower and inhibitor states
- **Real-time Streaming** - WebSocket API for low-latency updates (~10 FPS, 100ms)
- **Debug Monitor** - Live visualization at http://localhost:8765/debug

## Quick Start

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the service
python src/main.py
```

Then open http://localhost:8765/debug in your browser.

## API Usage

### WebSocket (Recommended)

```javascript
const ws = new WebSocket('ws://127.0.0.1:8765/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Champions:', data.champions);
    console.log('Jungle Camps:', data.jungleCamps);
    console.log('Structures:', data.structures);
};

// Request analysis (send any message to trigger)
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ requestTime: Date.now() }));
    }
}, 100);  // 10 FPS
```

### HTTP REST

```bash
# Single analysis
curl -X POST http://127.0.0.1:8765/analyze

# Health check
curl http://127.0.0.1:8765/health

# Get current calibration
curl http://127.0.0.1:8765/calibration
```

## Response Format

```json
{
    "timestamp": 1705789200000,
    "processingTimeMs": 45.2,
    "champions": [
        {
            "championName": "Ahri",
            "position": { "x": 45.5, "y": 62.3 },
            "team": "ORDER",
            "isPlayer": true,
            "confidence": 0.92
        }
    ],
    "jungleCamps": [
        {
            "type": "blue_buff",
            "position": { "x": 24.3, "y": 47.1 },
            "side": "ORDER",
            "status": "alive",
            "confidence": 0.88
        }
    ],
    "objectives": [
        {
            "type": "dragon",
            "position": { "x": 50.0, "y": 70.0 },
            "status": "alive",
            "confidence": 0.95
        }
    ],
    "structures": [
        {
            "structureType": "outer_turret",
            "position": { "x": 15.0, "y": 50.0 },
            "team": "ORDER",
            "lane": "top",
            "isAlive": true,
            "confidence": 1.0
        }
    ],
    "towers": {
        "ORDER": { "top": 3, "mid": 3, "bot": 3 },
        "CHAOS": { "top": 3, "mid": 3, "bot": 3 }
    },
    "metadata": {
        "minimapResolution": { "width": 420, "height": 420 },
        "detectionErrors": []
    }
}
```

**Coordinate System:** All positions are normalized to 0-100 range where (0,0) is top-left and (100,100) is bottom-right of the minimap.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and available endpoints |
| `/health` | GET | Health check with uptime and connection count |
| `/debug` | GET | Debug visualization page |
| `/analyze` | POST | Single HTTP analysis request |
| `/ws` | WebSocket | Real-time streaming (recommended) |
| `/calibration` | GET | Current minimap region settings |
| `/calibrate` | POST | Trigger minimap recalibration |
| `/screenshot` | GET | Latest minimap capture (JPEG) |
| `/templates/champions/{name}` | GET | Champion icon images |

## Configuration

Create a `.env` file in the project root:

```env
# Server
HOST=127.0.0.1
PORT=8765
DEBUG=true
LOG_LEVEL=INFO

# Detection Features
ENABLE_CHAMPION_DETECTION=true
ENABLE_JUNGLE_DETECTION=true
ENABLE_OBJECTIVE_DETECTION=true
ENABLE_TOWER_DETECTION=true

# Performance
TARGET_FPS=30
MAX_PROCESSING_TIME_MS=50

# Screen Capture
CAPTURE_METHOD=mss
AUTO_DETECT_MINIMAP=false

# Manual Minimap Calibration (if auto-detect is off)
MINIMAP_X=2134
MINIMAP_Y=1003
MINIMAP_WIDTH=420
MINIMAP_HEIGHT=420

# Serialization (json for debugging, msgpack for production)
SERIALIZATION_FORMAT=json
```

## Debug Monitor

Access http://localhost:8765/debug while the service is running to see:

- Live minimap visualization with detection overlays
- Champion icons with team-colored borders
- Jungle camp markers
- Structure positions
- Real-time JSON output
- FPS and processing time metrics

## Debug Logging

When `DEBUG=true`, the service logs JSON output to `logs/cv_output_*.jsonl`:
- Logs every 5 seconds
- Keeps maximum 3 log files
- Useful for reviewing detection output after stopping the service

## Project Structure

```
league-cv-service/
├── src/
│   ├── main.py              # FastAPI server, WebSocket endpoint
│   ├── config.py            # Pydantic settings
│   ├── api/
│   │   ├── routes.py        # HTTP endpoints
│   │   └── schemas.py       # Response models (Pydantic)
│   ├── capture/
│   │   └── screen.py        # MSS screen capture
│   ├── detection/
│   │   ├── base.py          # Base detector class
│   │   ├── champions.py     # Champion detection (Riot API + templates)
│   │   ├── jungle_camps.py  # Jungle camp detection
│   │   ├── objectives.py    # Dragon/Baron/Herald detection
│   │   └── structures.py    # Tower/inhibitor detection
│   └── processing/
│       └── pipeline.py      # Detection pipeline orchestrator
├── models/
│   └── templates/
│       ├── champions/       # Champion icon templates (.png)
│       ├── camps/           # Jungle camp templates
│       └── objectives/      # Objective templates
├── logs/                    # Debug JSON output (auto-created)
├── debug.html               # Debug visualization page
├── requirements.txt
└── .env
```

## How It Works

1. **Screen Capture** - MSS captures the minimap region at high speed (~2ms)
2. **Champion Detection** - Fetches current game champions from Riot's local API (127.0.0.1:2999), loads only those 10 templates, uses template matching
3. **Jungle Camps** - Template matching with orange color validation, position-based classification to fixed camp locations
4. **Objectives** - Template matching for Dragon, Baron, Herald icons
5. **Structures** - Fixed position detection for towers and inhibitors
6. **Output** - JSON via WebSocket or HTTP with normalized coordinates (0-100)

## Performance

| Metric | Typical Value |
|--------|---------------|
| Processing Time | 80-120ms |
| Effective FPS | ~10 |
| Screen Capture | ~2ms |

This is sufficient for macro game state tracking - champion positions, jungle timers, objective control. You don't need 60 FPS for strategic analysis.

## Requirements

- Python 3.10+
- Windows (for MSS screen capture)
- League of Legends running (for Riot API champion data)

## Building On This Service

This service is designed to be consumed by other applications:

```python
# Python client example
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    # Process game state...
    print(f"Detected {len(data['champions'])} champions")

ws = websocket.WebSocketApp("ws://127.0.0.1:8765/ws",
                            on_message=on_message)
ws.run_forever()
```

Potential applications:
- LLM-powered coaching overlay
- Jungle timer tracking
- Map awareness training tools
- Game replay analysis
- Stream overlays

## Troubleshooting

**Port in use:**
```bash
netstat -ano | findstr :8765
taskkill /PID <pid> /F
```

**No champion detection:**
- Riot API only available during active games

**Minimap not captured correctly:**
- Adjust `MINIMAP_X`, `MINIMAP_Y`, `MINIMAP_WIDTH`, `MINIMAP_HEIGHT` in `.env`
- Use `/calibrate` endpoint or debug monitor recalibration
- Game settings minimap size must be set to 59-62 range.

**Low FPS:**
- Disable unused detections in `.env`
- Reduce `TARGET_FPS`
