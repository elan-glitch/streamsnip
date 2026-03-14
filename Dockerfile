FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask flask-cors yt-dlp librosa numpy scipy openai-whisper opencv-python-headless

COPY . .

EXPOSE 5000

CMD ["python3", "backend_free.py"]
