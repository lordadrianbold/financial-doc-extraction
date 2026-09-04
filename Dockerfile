FROM python:3.12-slim

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ ./src/
COPY data/synthetic/synthetic_invoices.json ./data/synthetic/synthetic_invoices.json

# Explicit, targeted copy of service.py by its exact name — applied
# proactively this time, not reactively: the month-end-close-assistant
# project's deployment found a real, never-fully-explained issue where
# the broader `COPY src/ ./src/` above silently dropped this one file
# despite it genuinely existing on disk. Rather than risk rediscovering
# the same problem, this explicit copy is included from the start.
COPY src/service.py ./src/service.py

EXPOSE 8000

# Running via `python -m uvicorn` (not the bare `uvicorn` binary), from
# inside src/ directly — the exact, confirmed-working fix from the
# month-end-close-assistant deployment, applied here from the start
# rather than rediscovered: `python -m` explicitly adds the current
# working directory to Python's import path, which the bare binary
# invocation does not reliably do, and running from src/ avoids the
# package-style `src.service:app` import path issue entirely.
WORKDIR /app/src
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]




