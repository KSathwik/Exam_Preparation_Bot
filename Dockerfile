FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    gcc g++ libpoppler-cpp-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Run as a non-root user — a vulnerability triggered via a malicious
# uploaded PDF/DOCX (parsed by pdfplumber/python-docx) would otherwise
# execute with root privileges in-container. Directories the app writes to
# at runtime are chowned to this user; note that docker-compose.yml
# bind-mounts ./data, ./logs, ./uploads over these paths, so on the host
# those directories (or their contents) must also be writable by UID 1000,
# not just inside this image layer.
RUN mkdir -p logs cache uploads data/faiss_index \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
