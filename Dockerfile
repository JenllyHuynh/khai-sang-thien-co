FROM python:3.12-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

COPY src/ src/
COPY dashboard/ dashboard/

ENV PORT=8080
EXPOSE 8080

CMD streamlit run dashboard/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
