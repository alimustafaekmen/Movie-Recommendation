# Pinned to a patch release, so rebuilds get the same interpreter.
FROM python:3.12.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, for layer caching. The base image's pip has advisories, so a pinned
# newer pip is used; every download is hash-checked because the build runs as root.
COPY requirements.lock.txt .
RUN echo 'pip==26.2.2 --hash=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' > /tmp/pip.txt \
    && pip install --require-hashes -r /tmp/pip.txt \
    && rm /tmp/pip.txt \
    && pip install --require-hashes -r requirements.lock.txt

# Files stay owned by root, so the app user cannot change them.
COPY . .
RUN useradd --system --no-create-home appuser
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=4)"

# One worker: every worker would build its own copy of the model.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5001", "--access-logfile", "-", "app:app"]
