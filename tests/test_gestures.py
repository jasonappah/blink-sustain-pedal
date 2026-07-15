"""Unit tests for the wink state machine — no camera or MIDI required."""

from __future__ import annotations

import pytest

from blink_sustain_pedal.gestures import (
    BLINK_LEFT,
    BLINK_RIGHT,
    Eye,
    WinkDetector,
)


def bs(left: float, right: float) -> dict[str, float]:
    return {BLINK_LEFT: left, BLINK_RIGHT: right}


def feed(detector: WinkDetector, frames, dt_ms: float = 16.0):
    """Feed (left, right) pairs at dt_ms spacing; return list of readings."""
    out = []
    t = 0.0
    for left, right in frames:
        out.append(detector.update(bs(left, right), t))
        t += dt_ms
    return out


def test_left_wink_engages_and_holds():
    d = WinkDetector(eye=Eye.LEFT)
    # Left eye closed, right open, held for several frames.
    readings = feed(d, [(0.9, 0.05)] * 5)
    assert readings[0].down and readings[0].changed  # engages on first frame
    assert all(r.down for r in readings)
    assert not any(r.changed for r in readings[1:])  # no further edges while held


def test_full_blink_never_engages():
    d = WinkDetector(eye=Eye.EITHER)
    # Both eyes closed = involuntary blink → must be ignored.
    readings = feed(d, [(0.95, 0.92)] * 10)
    assert not any(r.down for r in readings)


def test_release_requires_debounce_hold():
    # release_ms=50 with 16ms frames: needs >=4 open frames (0,16,32,48 -> 48>=50? no; 64>=50 yes)
    d = WinkDetector(eye=Eye.EITHER, release_ms=50.0)
    frames = [(0.9, 0.05)] * 3 + [(0.05, 0.05)] * 5
    readings = feed(d, frames, dt_ms=16.0)
    assert readings[0].down  # engaged
    # First open frame (index 3) starts the release timer but pedal stays down.
    assert readings[3].down and not readings[3].changed
    # By the last frame the debounce window has elapsed and the pedal released.
    assert not readings[-1].down
    assert any(r.changed and not r.down for r in readings)


def test_brief_reopen_does_not_release():
    # Eye flickers open for one frame below release_ms → pedal must stay down.
    d = WinkDetector(eye=Eye.EITHER, release_ms=100.0)
    frames = [(0.9, 0.05), (0.9, 0.05), (0.1, 0.05), (0.9, 0.05), (0.9, 0.05)]
    readings = feed(d, frames, dt_ms=16.0)
    assert all(r.down for r in readings)


def test_hysteresis_between_on_and_off():
    # Score between off (0.3) and on (0.5) should neither engage nor, once engaged, release.
    d = WinkDetector(eye=Eye.EITHER, on_threshold=0.5, off_threshold=0.3, release_ms=0.0)
    # Never crosses on_threshold → never engages.
    assert not any(r.down for r in feed(d, [(0.4, 0.05)] * 5))

    d2 = WinkDetector(eye=Eye.EITHER, on_threshold=0.5, off_threshold=0.3, release_ms=0.0)
    # Engage, then drop into the hysteresis band (0.4) → stays down.
    readings = feed(d2, [(0.9, 0.05), (0.4, 0.05), (0.4, 0.05)])
    assert all(r.down for r in readings)


def test_open_threshold_blocks_engage_when_other_eye_partly_closed():
    # Winking eye high, but the "other" eye is above open_threshold → not a clean wink.
    d = WinkDetector(eye=Eye.LEFT, on_threshold=0.5, open_threshold=0.4)
    assert not any(r.down for r in feed(d, [(0.9, 0.6)] * 5))


def test_either_eye_accepts_right_wink():
    d = WinkDetector(eye=Eye.EITHER)
    readings = feed(d, [(0.05, 0.9)] * 3)
    assert readings[0].down
    assert readings[0].wink_score == pytest.approx(0.9)
    assert readings[0].other_score == pytest.approx(0.05)


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        WinkDetector(on_threshold=0.3, off_threshold=0.5)  # off > on
    with pytest.raises(ValueError):
        WinkDetector(open_threshold=1.5)
    with pytest.raises(ValueError):
        WinkDetector(release_ms=-1.0)
