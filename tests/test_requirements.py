"""The lockfiles match the exact pins in the requirements files they were made from, and the
Docker image installs nothing whose hash is unchecked."""

import re

import pytest
from conftest import REPO

PIN = re.compile(r"^([A-Za-z0-9._-]+)==(\S+)", re.MULTILINE)


def _pins(path):
    """Return {package name: version} of every name==version line in path."""
    text = (REPO / path).read_text(encoding="utf-8")
    return {name.lower(): version for name, version in PIN.findall(text)}


# A pin bumped in requirements.txt or requirements-dev.txt without regenerating its lockfile
# (README, Dependencies) would change nothing that CI or the Docker image installs.
@pytest.mark.parametrize(
    ("source", "lockfile"),
    [
        ("requirements.txt", "requirements.lock.txt"),
        ("requirements-dev.txt", "requirements-dev.lock.txt"),
        ("requirements.txt", "requirements-dev.lock.txt"),
    ],
)
def test_every_pin_is_in_the_lockfile(source, lockfile):
    locked = _pins(lockfile)

    for name, version in _pins(source).items():
        assert locked.get(name) == version, name


def test_every_pip_install_in_the_dockerfile_checks_hashes():
    # The build runs as root, so one unchecked download could change everything it installs.
    lines = (REPO / "Dockerfile").read_text(encoding="utf-8").splitlines()
    instructions = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    installs = re.findall(r"pip install [^&\n]*", instructions.replace("\\\n", " "))

    assert installs
    for install in installs:
        assert "--require-hashes" in install, install
