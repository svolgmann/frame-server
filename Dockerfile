FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY updater.py /app/updater.py

RUN pip install --no-cache-dir pillow numpy pysmb

EXPOSE 80

ENTRYPOINT ["python", "/app/updater.py"]
CMD ["--host", "0.0.0.0", "--port", "80", "--base-url", "http://localhost:80"]
