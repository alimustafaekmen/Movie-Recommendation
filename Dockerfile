# Pinned to a patch release, so a rebuild gets the same interpreter; CI also runs Python 3.12.
FROM python:3.12.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so a code change does not reinstall them. The pip that comes with the
# base image has known advisories, so a newer pinned pip installs them. The build runs as
# root, so every download, that pip included, is checked against its hash.
COPY requirements.lock.txt .
RUN echo 'pip==26.2.2 --hash=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' > /tmp/pip.txt \
    && pip install --require-hashes -r /tmp/pip.txt \
    && rm /tmp/pip.txt \
    && pip install --require-hashes -r requirements.lock.txt

# The code and the dataset CSVs stay owned by root: the app only reads them, so its user
# cannot change them. The user has no home directory, as nothing the app runs writes one.
COPY . .
RUN useradd --system --no-create-home appuser
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=4)"

# One worker: every worker would build its own copy of the model.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5001", "--access-logfile", "-", "app:app"]
