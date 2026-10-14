# Multi-mode production container for Spam Classifier
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    APP_MODE=streamlit

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY data/ data/
COPY models/ models/
COPY src/ src/
COPY outputs/ outputs/

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || curl --fail http://localhost:8000/health || exit 1

# Launch script supporting both Streamlit UI and FastAPI modes
CMD if [ "$APP_MODE" = "api" ]; then \
        uvicorn src.api:app --host 0.0.0.0 --port 8000; \
    else \
        streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0; \
    fi
