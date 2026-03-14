# StreamSnip — Free Local Clip Tool

Clip, trim and optionally transcribe any video supported by yt-dlp, entirely on your local machine.

## Setup (one-time)

```powershell
pip install flask flask-cors yt-dlp openai-whisper torch numpy scipy librosa opencv-python-headless
winget install ffmpeg          # or: brew install ffmpeg  (macOS)
```

## Run

```powershell
# 1. Start the backend (keep this terminal open)
python backend_free.py

# 2. Open the frontend in your browser
start streamsnip-free.html
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/info` | Fetch video metadata |
| POST | `/api/clip` | Start a clip job |
| GET  | `/api/job/:id` | Poll job progress |
| GET  | `/api/clips` | List all saved clips |
| DELETE | `/api/clips/:id` | Delete a clip |
| GET  | `/api/health` | Backend status |

## Clip request body

```json
{
  "url":        "https://www.youtube.com/watch?v=...",
  "start":      "1:30",
  "end":        "2:00",
  "format":     "",
  "transcribe": false
}
```

## Deploy online

- **Railway.app** — push to GitHub, auto-deploys
- **ngrok** — `ngrok http 5000` → instant public URL
- **Render / Fly.io** — free tiers available
