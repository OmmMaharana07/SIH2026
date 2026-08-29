# 🛰️ SatQuery AI — Deployment Guide (SIH 2026)

This document provides step-by-step instructions to deploy **SatQuery AI** across local environments, Docker containers, and cloud PaaS providers.

---

## 📋 Pre-Requisites & Dependencies

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## 🚀 Deployment Options

### Option 1: Production Local / Server WSGI (Waitress)
Waitress is a production-grade multi-threaded WSGI server for Windows and Linux.

```bash
# Start production server on port 5000 (8 worker threads)
python wsgi.py
```
To run on a custom port or host:
```bash
# PowerShell
$env:PORT="8080"
python wsgi.py

# Linux/Bash
PORT=8080 python wsgi.py
```

---

### Option 2: Docker Container Deployment
A multi-stage `Dockerfile` and `docker-compose.yml` are included.

```bash
# Build and start container in background
docker compose up --build -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```
The app will be accessible at `http://localhost:5000`.

---

### Option 3: Free Cloud Deployment (Render.com)
1. Push this repository to **GitHub** or **GitLab**.
2. Go to [dashboard.render.com](https://dashboard.render.com) and click **New Web Service**.
3. Select your repository.
4. Render will automatically detect `render.yaml` or `Procfile`:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 4 wsgi:app`
5. Click **Deploy**. Your app will get a public live URL: `https://satquery-ai.onrender.com`.

---

### Option 4: Railway.app / Fly.io
#### Railway:
1. Go to [railway.app](https://railway.app), click **New Project** -> **Deploy from GitHub Repo**.
2. Railway will automatically detect the `Procfile` / `Dockerfile` and start Gunicorn.

#### Fly.io:
```bash
fly launch
fly deploy
```

---

### Option 5: Hugging Face Spaces
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Docker** SDK.
2. Push the files in this directory to the Space repository.
3. The Space will build and launch automatically.

---

## 🔑 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PORT` | HTTP port for web server | `5000` |
| `HOST` | Bind host IP | `0.0.0.0` |
| `GEMINI_API_KEY` | *(Optional)* Google Gemini Vision API Key | *(Offline mode if empty)* |
| `DATASET_PATH` | Path to benchmark images directory | `../archive/data` or `./data` |
