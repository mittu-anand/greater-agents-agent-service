FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Agent config injected at runtime via env vars
# AGENT_ID, AGENT_NAME, SYSTEM_PROMPT, LLM, API_KEY, TEMPERATURE, PORT
ENV PORT=8001

EXPOSE 8001

CMD ["python", "main.py"]
