"""The lockfile matches the exact pins in requirements.txt, and the Docker image installs
nothing whose hash is unchecked."""

import re

from conftest import REPO

PIN = re.compile(r"^([A-Za-z0-9._-]+)==(\S+)", re.MULTILINE)


def _pins(path):
    """Return {package name: version} of every name==version line in path."""
    text = (REPO / path).read_text(encoding="utf-8")
    return {name.lower(): version for name, version in PIN.findall(text)}


# A pin bumped in requirements.txt without regenerating the lockfile (README, Dependencies)
# would change nothing that CI or the Docker image installs.
def test_every_pin_is_in_the_lockfile():
    locked = _pins("requirements.lock.txt")

    for name, version in _pins("requirements.txt").items():
        assert locked.get(name) == version, name


def test_every_pip_install_in_the_dockerfile_checks_hashes():
    # The build runs as root, so one unchecked download could change everything it installs.
    lines = (REPO / "Dockerfile").read_text(encoding="utf-8").splitlines()
    instructions = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    installs = re.findall(r"pip install [^&\n]*", instructions.replace("\\\n", " "))

    assert installs
    for install in installs:
        assert "--require-hashes" in install, install
