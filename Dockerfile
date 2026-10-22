# The exact Python of the CI and the development setup, so a rebuild gets the same interpreter.
FROM python:3.12.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# The dataset CSVs are copied in as well; the app only reads them and writes no files, so the
# files stay owned by root and the app user cannot change them.
COPY . .
RUN useradd --create-home appuser
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=4)"

# One worker: every worker would build its own copy of the model.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5001", "--access-logfile", "-", "app:app"]
