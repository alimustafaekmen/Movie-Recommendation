"""The lockfiles match the exact pins in the requirements files they were made from."""

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
