# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (Hugging Face uses 7860 by default, but we'll use the PORT env var)
EXPOSE 7860

# Run the application
# We use gunicorn for production stability
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
