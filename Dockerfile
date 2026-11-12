FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs at runtime.
COPY predict.py .
COPY model/ ./model/

EXPOSE 9696

# Render/Heroku provide $PORT; locally it falls back to 9696.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-9696} --workers 2 --timeout 60 predict:app"]
