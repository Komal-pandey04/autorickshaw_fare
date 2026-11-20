FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# dependencies first, so this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# then the code, config and the champion model
COPY params.yaml .
COPY src/ src/
COPY app/ app/
COPY models/ models/

# run as a non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
