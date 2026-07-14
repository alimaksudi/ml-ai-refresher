FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WINE_MODEL_DIR=/app/artifacts

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY artifacts /app/artifacts

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "wine_classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
