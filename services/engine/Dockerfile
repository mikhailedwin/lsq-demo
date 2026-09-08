FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY qav_engine ./qav_engine
RUN uv pip install --system --no-cache . \
 && python -m qav_engine.worker download-files
CMD ["python", "-m", "qav_engine.worker", "start"]
