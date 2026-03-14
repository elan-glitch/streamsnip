"""
StreamSnip Backend (Free / Local)
==================================
Dependencies:
    pip install flask flask-cors yt-dlp openai-whisper torch numpy scipy librosa opencv-python-headless
    ffmpeg must be on PATH
"""

import os
import re
import uuid
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── Optional heavy imports ──────────────────────────────────────────────────
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

# ── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR   = Path(__file__).parent
CLIPS_DIR  = BASE_DIR / "clips"
AUDIO_DIR  = BASE_DIR / "audio"
THUMB_DIR  = BASE_DIR / "thumbnails"

for d in (CLIPS_DIR, AUDIO_DIR, THUMB_DIR):
    d.mkdir(exist_ok=True)

# In-memory job store  {job_id: {...}}
JOBS: dict[str, dict] = {}
_whisper_model = None
_whisper_lock  = threading.Lock()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_whisper_model(size="base"):
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None and WHISPER_AVAILABLE:
            _whisper_model = whisper.load_model(size)
    return _whisper_model


def _job(job_id: str):
    return JOBS.setdefault(job_id, {
        "id":       job_id,
        "status":   "pending",
        "progress": 0,
        "message":  "",
        "result":   None,
        "error":    None,
        "created":  datetime.utcnow().isoformat(),
    })


def _parse_time(t: str) -> float:
    """Accept  ss,  mm:ss,  hh:mm:ss  or  float strings."""
    t = t.strip()
    parts = t.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        raise ValueError(f"Cannot parse time: {t!r}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        **kwargs
    )


# ── Video info ───────────────────────────────────────────────────────────────

@app.route("/api/info", methods=["POST"])
def get_info():
    if not YTDLP_AVAILABLE:
        return jsonify({"error": "yt-dlp not installed"}), 500

    data = request.get_json(force=True)
    url  = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    ydl_opts = {
        "quiet":           True,
        "skip_download":   True,
        "forcejson":       True,
        "no_warnings":     True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return jsonify({
            "title":       info.get("title"),
            "duration":    info.get("duration"),
            "thumbnail":   info.get("thumbnail"),
            "uploader":    info.get("uploader"),
            "view_count":  info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "webpage_url": info.get("webpage_url", url),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Clip creation ─────────────────────────────────────────────────────────────

def _clip_worker(job_id: str, url: str, start: float, end: float,
                 format_id: str, transcribe: bool):
    job = _job(job_id)

    try:
        tmp = Path(tempfile.mkdtemp(prefix="streamsnip_"))

        # ── 1. Download source ──────────────────────────────────────────────
        job.update(status="downloading", progress=5, message="Downloading video…")
        raw_path = tmp / "source.%(ext)s"

        ydl_opts: dict = {
            "quiet":    True,
            "outtmpl":  str(raw_path),
            "format":   format_id or "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "retries":  3,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # find whatever file was written
        src_files = list(tmp.glob("source.*"))
        if not src_files:
            raise FileNotFoundError("Download produced no output file.")
        src = src_files[0]

        # ── 2. Trim ─────────────────────────────────────────────────────────
        job.update(status="clipping", progress=40, message="Trimming clip…")
        clip_name = f"{job_id}.mp4"
        clip_path = CLIPS_DIR / clip_name
        duration  = end - start

        _run([
            "ffmpeg", "-y",
            "-ss",  str(start),
            "-i",   str(src),
            "-t",   str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            str(clip_path),
        ])

        # ── 3. Thumbnail ─────────────────────────────────────────────────────
        job.update(progress=60, message="Generating thumbnail…")
        thumb_path = THUMB_DIR / f"{job_id}.jpg"
        try:
            _run([
                "ffmpeg", "-y",
                "-ss",    str(start + duration / 2),
                "-i",     str(src),
                "-vframes", "1",
                "-q:v",   "2",
                str(thumb_path),
            ])
        except Exception:
            thumb_path = None

        # ── 4. Transcribe ────────────────────────────────────────────────────
        transcript = None
        if transcribe and WHISPER_AVAILABLE:
            job.update(progress=70, message="Transcribing audio (Whisper)…")
            audio_path = AUDIO_DIR / f"{job_id}.wav"
            # Extract mono 16 kHz wav
            _run([
                "ffmpeg", "-y",
                "-i",  str(clip_path),
                "-ar", "16000", "-ac", "1",
                str(audio_path),
            ])
            model = _get_whisper_model()
            result = model.transcribe(str(audio_path), fp16=False)
            transcript = result.get("text", "")
            segments   = [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ]
        else:
            segments = []

        # ── 5. Done ──────────────────────────────────────────────────────────
        shutil.rmtree(tmp, ignore_errors=True)

        job.update(
            status   = "done",
            progress = 100,
            message  = "Clip ready!",
            result   = {
                "clip_url":   f"/clips/{clip_name}",
                "thumb_url":  f"/thumbnails/{job_id}.jpg" if thumb_path else None,
                "duration":   duration,
                "start":      start,
                "end":        end,
                "transcript": transcript,
                "segments":   segments,
            },
        )

    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        job.update(status="error", progress=0, message=str(exc), error=str(exc))


@app.route("/api/clip", methods=["POST"])
def create_clip():
    data = request.get_json(force=True) or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    try:
        start = _parse_time(str(data.get("start", "0")))
        end   = _parse_time(str(data.get("end",   "30")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if end <= start:
        return jsonify({"error": "end must be after start"}), 400
    if (end - start) > 600:
        return jsonify({"error": "Clip too long (max 10 min)"}), 400

    job_id    = uuid.uuid4().hex
    format_id = data.get("format", "")
    transcribe = bool(data.get("transcribe", False))

    _job(job_id)
    t = threading.Thread(
        target=_clip_worker,
        args=(job_id, url, start, end, format_id, transcribe),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id}), 202


# ── Job status ────────────────────────────────────────────────────────────────

@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


# ── Serve clips & thumbnails ──────────────────────────────────────────────────

@app.route("/clips/<path:filename>")
def serve_clip(filename):
    p = CLIPS_DIR / filename
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p, mimetype="video/mp4")


@app.route("/thumbnails/<path:filename>")
def serve_thumb(filename):
    p = THUMB_DIR / filename
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p, mimetype="image/jpeg")


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "whisper":          WHISPER_AVAILABLE,
        "yt_dlp":           YTDLP_AVAILABLE,
        "ffmpeg":           shutil.which("ffmpeg") is not None,
        "clips_dir":        str(CLIPS_DIR),
        "active_jobs":      len([j for j in JOBS.values() if j["status"] not in ("done","error")]),
    })


# ── Clip list ─────────────────────────────────────────────────────────────────

@app.route("/api/clips", methods=["GET"])
def list_clips():
    clips = []
    for j in JOBS.values():
        if j["status"] == "done" and j.get("result"):
            clips.append({
                "id":        j["id"],
                "created":   j["created"],
                **j["result"],
            })
    clips.sort(key=lambda c: c["created"], reverse=True)
    return jsonify(clips)


# ── Delete clip ───────────────────────────────────────────────────────────────

@app.route("/api/clips/<job_id>", methods=["DELETE"])
def delete_clip(job_id: str):
    job = JOBS.pop(job_id, None)
    if not job:
        return jsonify({"error": "not found"}), 404
    for f in [CLIPS_DIR / f"{job_id}.mp4",
              THUMB_DIR  / f"{job_id}.jpg",
              AUDIO_DIR  / f"{job_id}.wav"]:
        f.unlink(missing_ok=True)
    return jsonify({"deleted": job_id})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎬  StreamSnip Backend  —  http://localhost:5000")
    print(f"   yt-dlp   : {'✓' if YTDLP_AVAILABLE else '✗ (pip install yt-dlp)'}")
    print(f"   Whisper  : {'✓' if WHISPER_AVAILABLE else '✗ (pip install openai-whisper)'}")
    print(f"   ffmpeg   : {'✓' if shutil.which('ffmpeg') else '✗ (install ffmpeg and add to PATH)'}")
    print()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=5000)
