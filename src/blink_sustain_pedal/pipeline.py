"""The hot loop: latest frame → blendshapes → gesture → MIDI.

Skips frames it has already processed (via the camera's ``seq``) and sleeps only
when idle, so latency stays as low as the camera allows while CPU stays sane.
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from .camera import CameraStream
from .gestures import GestureDetector
from .landmarker import FaceBlendshapes
from .midi import SustainOutput

_IDLE_SLEEP_S = 0.001  # only when no new frame is available
_CALIBRATE_HZ = 20.0


def run(
    camera: CameraStream,
    landmarker: FaceBlendshapes,
    detector: GestureDetector,
    midi: SustainOutput | None,
    *,
    calibrate: bool = False,
    stop: Callable[[], bool] | None = None,
) -> None:
    start = time.monotonic()
    last_seq = -1
    last_ts = -1.0
    last_print = 0.0

    try:
        while stop is None or not stop():
            seq, frame = camera.read()
            if frame is None or seq == last_seq:
                time.sleep(_IDLE_SLEEP_S)
                continue
            last_seq = seq

            # VIDEO mode requires strictly increasing timestamps.
            ts = (time.monotonic() - start) * 1000.0
            if ts <= last_ts:
                ts = last_ts + 1.0
            last_ts = ts

            blendshapes = landmarker.detect(frame, ts)
            reading = detector.update(blendshapes, ts)

            if reading.changed and midi is not None:
                midi.set(reading.down)

            if calibrate:
                now = time.monotonic()
                if now - last_print >= 1.0 / _CALIBRATE_HZ:
                    _print_reading(reading, no_face=not blendshapes)
                    last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        if calibrate:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _print_reading(reading, *, no_face: bool) -> None:
    state = "DOWN" if reading.down else "up  "
    face = "" if not no_face else "  (no face)"
    sys.stdout.write(
        f"\rL {reading.left:0.2f}  R {reading.right:0.2f}  "
        f"| wink {reading.wink_score:0.2f}  other {reading.other_score:0.2f}  "
        f"| pedal {state}{face}   "
    )
    sys.stdout.flush()
