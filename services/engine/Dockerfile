FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
# the face package (CPU parts) is a dependency of the engine
COPY services/face /app/face
COPY services/engine/pyproject.toml /app/engine/
COPY services/engine/qav_engine /app/engine/qav_engine
RUN uv pip install --system --no-cache /app/face /app/engine \
 && cd /app/engine && python -m qav_engine.worker download-files
WORKDIR /app/engine
CMD ["python", "-m", "qav_engine.worker", "start"]
