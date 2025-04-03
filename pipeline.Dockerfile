FROM python:3.10-slim
ARG REPOCACHE_HOST

COPY <<EOF /etc/pip.conf
[global]
index-url = https://${REPOCACHE_HOST}/artifactory/api/pypi/pypi/simple
EOF

# Set working directory
WORKDIR /app

# Create a non-root user for security
RUN groupadd -r -g 1000 appuser && useradd -r -g appuser -u 1000appuser

# Create error.html for the web server error page
RUN echo '<!DOCTYPE html><html><head><title>Error</title></head><body><h1>Error</h1><p>The requested page could not be found.</p></body></html>' > error.html

# Install curl for the healthcheck
RUN apt-get update && apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port that the server runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy only the files needed for the web server
COPY server.py .
COPY index.html .

# Command to run the server
CMD ["python", "server.py"]
