FROM python:3.10

WORKDIR /app

# Copy project first (better caching behavior)
COPY . .

# Install backend dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Render exposes PORT dynamically
EXPOSE 10000

# Start FastAPI app
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]
