# --- Build stage ---
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM python:3.12-slim

RUN groupadd -r vvalley && useradd -r -g vvalley -d /app -s /sbin/nologin vvalley

WORKDIR /app
COPY --from=builder /install /usr/local

COPY packages/ packages/
COPY apps/ apps/
COPY assets/templates/ assets/templates/

RUN mkdir -p /app/data && chown -R vvalley:vvalley /app

USER vvalley

ENV PYTHONUNBUFFERED=1
ENV VVALLEY_AUTOSTART_TOWN_SCHEDULER=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["uvicorn", "apps.api.vvalley_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
