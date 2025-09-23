FROM python:3.9-slim

# # Create a non-root user with specific UID/GID
# RUN groupadd -g 1000 appuser && \
#     useradd -r -u 1000 -g appuser appuser

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/. ./
# RUN chown -R appuser:appuser /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
# USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
