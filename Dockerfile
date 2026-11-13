# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environmental variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and the fractal-core package
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
RUN pip install --no-cache-dir --no-deps .

# Create directories for runtime data
RUN mkdir -p src/fractal_server/uploads \
             src/fractal_server/downloads \
             src/fractal_server/tenants \
             src/fractal_server/global_model \
             data/downloads

# Expose the port the app runs on
EXPOSE 5000

# Single worker with multiple threads keeps in-process tenant locks consistent
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--timeout", "120", "src.fractal_server.server:app"]
