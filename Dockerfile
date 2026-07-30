FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY templates ./templates

ENV DATA_DIR=/data \
    PORT=41733

EXPOSE 41733
CMD ["python", "server.py"]
