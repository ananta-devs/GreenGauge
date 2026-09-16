# 🌱 Green AI Experiment Tracker

A lightweight tracking and monitoring framework designed to quantify, analyze, and minimize compute energy consumption and carbon emissions during AI and ML experiments.

---

## 📁 Project Structure

```text
green-ai-experiment-tracker/
│
├── backend/
│   ├── app/           # Core application configuration and utilities
│   ├── experiments/   # Backend experiment orchestration & management
│   ├── services/      # Business logic (energy tracking, emissions computation)
│   ├── models/        # Data schemas and database/domain models
│   └── main.py        # Application entrypoint (FastAPI)
│
├── experiments/       # Experiment scripts, configurations, and benchmarks
├── results/           # Output metrics, reports, and emissions logs
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### 2. Setup Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Backend Server
```bash
python -m uvicorn backend.main:app --reload
```
API documentation will be accessible at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 🌿 Core Features

- **Energy & Carbon Tracking**: Track GPU/CPU wattage and calculate real-time carbon footprint.
- **Experiment Benchmarking**: Compare efficiency across model architectures and training pipelines.
- **Modular Architecture**: Clean separation between tracking backend services, data models, and experiment runs.
