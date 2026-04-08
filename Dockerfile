FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Start the OpenEnv server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]