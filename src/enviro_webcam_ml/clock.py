from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClockCheckResult:
    ok: bool
    reason: str
    wall_delta_seconds: float | None = None
    monotonic_delta_seconds: float | None = None
    drift_seconds: float | None = None


class ClockSanityChecker:
    """Detect wall-clock behavior that would make capture timestamps untrustworthy.

    The collector schedules work with ``time.monotonic()``, which is independent
    from the system date/time. Captures are timestamped with wall-clock UTC time.
    Comparing the two lets us notice a frozen clock, a backward jump, or a large
    manual/NTP correction.
    """

    def __init__(
        self,
        *,
        max_drift_seconds: float,
        max_backward_seconds: float,
    ) -> None:
        self.max_drift_seconds = max_drift_seconds
        self.max_backward_seconds = max_backward_seconds
        self._last_wall_time: datetime | None = None
        self._last_monotonic_time: float | None = None

    def check(self, *, wall_time: datetime, monotonic_time: float) -> ClockCheckResult:
        if self._last_wall_time is None or self._last_monotonic_time is None:
            self._remember(wall_time, monotonic_time)
            return ClockCheckResult(ok=True, reason="initial clock sample")

        wall_delta = (wall_time - self._last_wall_time).total_seconds()
        monotonic_delta = monotonic_time - self._last_monotonic_time
        drift = abs(wall_delta - monotonic_delta)

        if monotonic_delta < 0:
            result = ClockCheckResult(
                ok=False,
                reason="monotonic clock moved backward",
                wall_delta_seconds=wall_delta,
                monotonic_delta_seconds=monotonic_delta,
                drift_seconds=drift,
            )
        elif wall_delta < -self.max_backward_seconds:
            result = ClockCheckResult(
                ok=False,
                reason="system wall clock moved backward",
                wall_delta_seconds=wall_delta,
                monotonic_delta_seconds=monotonic_delta,
                drift_seconds=drift,
            )
        elif drift > self.max_drift_seconds:
            result = ClockCheckResult(
                ok=False,
                reason="system wall clock drifted relative to monotonic time",
                wall_delta_seconds=wall_delta,
                monotonic_delta_seconds=monotonic_delta,
                drift_seconds=drift,
            )
        else:
            result = ClockCheckResult(
                ok=True,
                reason="clock sane",
                wall_delta_seconds=wall_delta,
                monotonic_delta_seconds=monotonic_delta,
                drift_seconds=drift,
            )

        # Always remember the latest sample. If the user manually fixes the
        # clock, we skip the suspicious cycle once, then recover on the next
        # sane sample instead of staying anchored to an old bad timestamp.
        self._remember(wall_time, monotonic_time)
        return result

    def _remember(self, wall_time: datetime, monotonic_time: float) -> None:
        self._last_wall_time = wall_time
        self._last_monotonic_time = monotonic_time
