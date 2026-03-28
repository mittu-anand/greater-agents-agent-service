FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update -qq && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

ENV PORT=8001
EXPOSE 8001

CMD ["python", "main.py"]
