FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala las dependencias primero para que Docker pueda reutilizar esta capa.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia solo lo que el servicio necesita en tiempo de ejecución.
COPY app.py .
COPY model/ ./model/

EXPOSE 9696

# La mayoría de plataformas entregan $PORT; localmente se usa 9696 por defecto.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-9696} --workers 2 --timeout 60 app:app"]
