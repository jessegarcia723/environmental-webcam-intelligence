from datetime import datetime, timedelta, timezone

from enviro_webcam_ml.clock import ClockSanityChecker


def test_clock_sanity_accepts_normal_elapsed_time() -> None:
    checker = ClockSanityChecker(max_drift_seconds=5, max_backward_seconds=1)
    start = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)

    assert checker.check(wall_time=start, monotonic_time=100).ok is True
    result = checker.check(wall_time=start + timedelta(seconds=60), monotonic_time=160)

    assert result.ok is True
    assert result.drift_seconds == 0


def test_clock_sanity_rejects_frozen_wall_clock() -> None:
    checker = ClockSanityChecker(max_drift_seconds=5, max_backward_seconds=1)
    start = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)

    checker.check(wall_time=start, monotonic_time=100)
    result = checker.check(wall_time=start, monotonic_time=160)

    assert result.ok is False
    assert "drifted" in result.reason


def test_clock_sanity_rejects_backward_wall_clock() -> None:
    checker = ClockSanityChecker(max_drift_seconds=120, max_backward_seconds=1)
    start = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)

    checker.check(wall_time=start, monotonic_time=100)
    result = checker.check(wall_time=start - timedelta(seconds=10), monotonic_time=101)

    assert result.ok is False
    assert "backward" in result.reason


def test_clock_sanity_recovers_after_one_bad_sample() -> None:
    checker = ClockSanityChecker(max_drift_seconds=5, max_backward_seconds=1)
    start = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)

    checker.check(wall_time=start, monotonic_time=100)
    bad = checker.check(wall_time=start + timedelta(hours=2), monotonic_time=101)
    recovered = checker.check(wall_time=start + timedelta(hours=2, seconds=60), monotonic_time=161)

    assert bad.ok is False
    assert recovered.ok is True
