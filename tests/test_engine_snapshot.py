import pytest
from snapshot_engine import BASELINE_PATH, build_snapshot, diff_against_baseline

pytestmark = pytest.mark.real_dataset


def test_engine_output_matches_baseline(engine, details):
    assert BASELINE_PATH.exists(), "no baseline yet: run python tests/snapshot_engine.py --update"

    diff = diff_against_baseline(build_snapshot(engine, details))

    if diff:
        pytest.fail(
            "Engine output differs from tests/snapshots/engine_baseline.json. If the change is "
            "intended, run python tests/snapshot_engine.py --update and review the diff.\n\n"
            + diff,
            pytrace=False,
        )
