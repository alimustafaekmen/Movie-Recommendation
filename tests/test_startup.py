"""Starting the app with a broken setup: clear log lines or a clear exit, never a traceback."""

import logging
import os
import subprocess
import sys

import pytest
from conftest import FIXTURE_DATASET_DIR, REPO
from flask import Response


def _run(code, **env):
    """Run Python code in a fresh process from the repo, as gunicorn or python app.py would."""
    environment = {
        **os.environ,
        "TMDB_API_KEY": "",
        "FLASK_SECRET_KEY": "test-secret-key",
        "DATASET_DIR": str(FIXTURE_DATASET_DIR),
        "LOG_LEVEL": "INFO",
        **env,
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_broken_setup_starts_with_one_clear_line_per_problem(tmp_path):
    missing = tmp_path / "nowhere"

    result = _run(
        "import app; print(app.app.test_client().get('/health').status_code)",
        DATASET_DIR=str(missing),
        LOG_LEVEL="verbose",
        FLASK_SECRET_KEY="your_secret_key_here",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "503"
    log = result.stderr
    assert "Traceback" not in log
    assert "LOG_LEVEL 'VERBOSE' is not a logging level; using INFO" in log
    assert f"Recommendation engine failed to start: Movies CSV not found: {missing}" in log
    assert "DATASET_DIR" in log
    assert "TMDB_API_KEY is not set" in log
    assert "FLASK_SECRET_KEY is not set" in log


def test_a_csv_without_the_expected_columns_is_reported_without_a_traceback(tmp_path):
    for name in ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv"):
        (tmp_path / name).write_text("a,b\n1,2\n", encoding="utf-8")

    result = _run("import app", DATASET_DIR=str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "Recommendation engine failed to start: " in result.stderr


def test_a_working_setup_logs_no_warning():
    result = _run("import app", TMDB_API_KEY="abc123")

    assert result.returncode == 0, result.stderr
    assert " WARNING " not in result.stderr
    assert " ERROR " not in result.stderr


def test_an_invalid_port_stops_the_dev_server_with_a_clear_message():
    result = _run("import runpy; runpy.run_path('app.py', run_name='__main__')", PORT="abc")

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "PORT must be a number from 1 to 65535, not 'abc'" in result.stderr


@pytest.mark.parametrize("raw", ["0", "70000", "-1", "5001.0", "٥٠٠١", "5 001"])
def test_port_must_be_a_plain_number_in_range(app_module, raw):
    with pytest.raises(SystemExit, match="PORT must be a number from 1 to 65535"):
        app_module._port(raw)


def test_port_accepts_the_usual_values(app_module):
    assert [app_module._port(raw) for raw in ("1", "5001", "65535")] == [1, 5001, 65535]


@pytest.mark.parametrize(
    ("path", "logged"),
    [("/movie/27205", True), ("/static/css/base.css", False), ("/health", False)],
)
def test_the_dev_server_logs_page_requests_only(app_module, caplog, path, logged):
    with app_module.app.test_request_context(path), caplog.at_level(logging.INFO):
        app_module._start_timer()
        app_module._log_request(Response(status=200))

    assert (f"GET {path} 200" in caplog.text) is logged
