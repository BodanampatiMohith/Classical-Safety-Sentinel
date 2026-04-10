# Classical Safety Sentinel

Hybrid deep-classical near-miss detection for urban traffic scenes.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-Frontend-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-Build-646cff.svg)](https://vitejs.dev/)

[![Architecture](https://img.shields.io/badge/View-Architecture-1f6feb?style=for-the-badge)](./ARCHITECTURE.md)
[![Algorithm](https://img.shields.io/badge/View-Algorithm-7a3cff?style=for-the-badge)](./ALGORITHM.md)
[![Backend API](https://img.shields.io/badge/OpenAPI-FastAPI-0ea5e9?style=for-the-badge)](http://127.0.0.1:8000/docs)
[![Frontend](https://img.shields.io/badge/Frontend-React_UI-14b8a6?style=for-the-badge)](./frontend)

## Project Status

This project is under active development. The architecture and thresholds may change as model tuning and validation progress.

## What It Does

- Detects and tracks traffic actors from video frames.
- Extracts spatial, temporal, and interaction safety features.
- Runs a hybrid risk model:
  - Deep path: temporal anomaly signal.
  - Classical path: rule-based safety checks.
- Fuses both paths into a final risk score and label (`SAFE`, `WARNING`, `CRITICAL`).
- Serves results via FastAPI and displays them in a React dashboard.

## Tech Stack

| Layer | Languages / Tools |
|---|---|
| Backend | Python, FastAPI, PyTorch |
| Vision + Logic | Python (YOLO pipeline, tracking, feature engineering, rule engine) |
| Frontend | JavaScript, React, Vite, HTML, CSS |
| Docs | Markdown |

## Repository Structure

```text
.
|-- core/
|   |-- perception.py
|   |-- features.py
|   `-- decision.py
|-- frontend/
|   |-- src/
|   |   |-- App.jsx
|   |   |-- App.css
|   |   `-- main.jsx
|   |-- index.html
|   |-- package.json
|   `-- vite.config.js
|-- main.py
|-- pipeline.py
|-- ARCHITECTURE.md
|-- ALGORITHM.md
`-- requirements.txt
```

## Quick Start

### 1) Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend endpoints:
- API root: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend app:
- `http://127.0.0.1:5173/` (default Vite port)

## Documentation

- [Architecture](./ARCHITECTURE.md): system components, data flow, deployment view.
- [Algorithm](./ALGORITHM.md): hybrid scoring logic and decision thresholds.

## Notes

- Dataset and large model artifacts are intentionally excluded from Git history.
- Current thresholds are practical defaults and should be calibrated for each camera setup.
