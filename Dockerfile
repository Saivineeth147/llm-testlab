# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY . .
WORKDIR /app/playground/frontend
RUN npm install
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies (as root before switching user - done in base image effectively)
# But we need to switch back to root to install system deps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from root (requires build context to be set to root '.')
COPY --chown=user . .

# Install main package in editable mode
# Switch to user for pip installs to avoid permission issues
USER user
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Install playground dependencies
RUN pip install --no-cache-dir -r playground/requirements.txt
RUN pip install --no-cache-dir groq huggingface_hub

# Copy built frontend from stage 1
COPY --from=frontend-builder --chown=user /app/playground/frontend/dist $HOME/app/playground/frontend/dist

# Set environment variables
ENV PORT=7860

# Expose port 7860 (HuggingFace default)
EXPOSE 7860

# Run server (relative to /home/user/app root)
CMD ["python", "playground/server.py"]
