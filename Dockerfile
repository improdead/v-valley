FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/ packages/
COPY apps/ apps/
COPY assets/templates/ assets/templates/
COPY data/ data/

ENV PYTHONUNBUFFERED=1
ENV VVALLEY_AUTOSTART_TOWN_SCHEDULER=true

EXPOSE 8080

CMD ["uvicorn", "apps.api.vvalley_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
