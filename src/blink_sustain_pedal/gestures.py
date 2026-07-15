"""Turn face blendshape scores into a pedal on/off decision.

Pure logic — no camera, no MIDI, no MediaPipe imports — so it is trivially
unit-testable. The pipeline feeds this a ``{blendshape_name: score}`` dict plus a
monotonic millisecond timestamp on every frame; it returns the resulting pedal state.

Wink-vs-blink discrimination is the whole trick: a *wink* is one eye closed while
the other stays open, whereas an involuntary *blink* closes both. We only engage
when the "winking" eye's score is high AND the other eye's score stays low, so
normal blinking never fires the pedal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol

# MediaPipe blendshape category names (ARKit convention: Left/Right are the
# subject's own anatomical eyes, independent of the mirrored camera image).
BLINK_LEFT = "eyeBlinkLeft"
BLINK_RIGHT = "eyeBlinkRight"


class Eye(str, Enum):
    """Which eye engages the pedal. ``EITHER`` accepts a wink of whichever eye."""

    LEFT = "left"
    RIGHT = "right"
    EITHER = "either"


@dataclass(frozen=True)
class WinkReading:
    """Result of one detector update — also drives the calibration readout."""

    left: float          # raw eyeBlinkLeft score
    right: float         # raw eyeBlinkRight score
    wink_score: float    # score of the eye currently treated as "winking"
    other_score: float   # score of the other eye (must stay low to count as a wink)
    down: bool           # pedal state after this update
    changed: bool        # did the pedal state flip on this update


class GestureDetector(Protocol):
    """Minimal contract so other gestures (jaw open, brow raise, …) can drop in."""

    def update(self, blendshapes: Mapping[str, float], timestamp_ms: float) -> WinkReading: ...

    def reset(self) -> None: ...


class WinkDetector:
    """Momentary sustain: pedal is down while a single-eye wink is held.

    Hysteresis (``on_threshold`` > ``off_threshold``) stops the pedal chattering
    when a score hovers near a single threshold. ``open_threshold`` gates
    *engagement* only — the other eye must be open to distinguish a wink from a
    blink — while release depends solely on the winking eye reopening, so a full
    blink mid-hold does not drop an already-engaged pedal. ``release_ms`` holds the
    pedal briefly after the eye reopens to smooth out flicker.
    """

    def __init__(
        self,
        eye: Eye = Eye.EITHER,
        on_threshold: float = 0.5,
        off_threshold: float = 0.3,
        open_threshold: float = 0.4,
        release_ms: float = 60.0,
    ) -> None:
        if not 0.0 <= off_threshold <= on_threshold <= 1.0:
            raise ValueError("require 0 <= off_threshold <= on_threshold <= 1")
        if not 0.0 <= open_threshold <= 1.0:
            raise ValueError("open_threshold must be within [0, 1]")
        if release_ms < 0:
            raise ValueError("release_ms must be non-negative")
        self.eye = eye
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.open_threshold = open_threshold
        self.release_ms = release_ms
        self._down = False
        self._release_since: float | None = None

    @property
    def down(self) -> bool:
        return self._down

    def reset(self) -> None:
        self._down = False
        self._release_since = None

    def _scores(self, blendshapes: Mapping[str, float]) -> tuple[float, float, float, float]:
        """Return (left, right, wink_score, other_score) for the configured eye."""
        left = float(blendshapes.get(BLINK_LEFT, 0.0))
        right = float(blendshapes.get(BLINK_RIGHT, 0.0))
        if self.eye is Eye.LEFT:
            return left, right, left, right
        if self.eye is Eye.RIGHT:
            return left, right, right, left
        # EITHER: the more-closed eye is the candidate wink.
        if left >= right:
            return left, right, left, right
        return left, right, right, left

    def update(self, blendshapes: Mapping[str, float], timestamp_ms: float) -> WinkReading:
        left, right, wink, other = self._scores(blendshapes)
        changed = False

        if not self._down:
            # Engage only on a genuine wink: target eye closed, other eye open.
            if wink >= self.on_threshold and other <= self.open_threshold:
                self._down = True
                self._release_since = None
                changed = True
        else:
            # Release once the winking eye reopens, held for release_ms to debounce.
            if wink < self.off_threshold:
                if self._release_since is None:
                    self._release_since = timestamp_ms
                elif timestamp_ms - self._release_since >= self.release_ms:
                    self._down = False
                    self._release_since = None
                    changed = True
            else:
                self._release_since = None  # eye still (mostly) closed → cancel release

        return WinkReading(left, right, wink, other, self._down, changed)
