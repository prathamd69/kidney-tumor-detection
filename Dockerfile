FROM python:3.10-slim

WORKDIR /app

# Install native system build tools safely
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY templates/ ./templates/

COPY litemodels/ ./litemodels/

COPY app.py .
COPY class_descriptions.json .
COPY params.yaml .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]