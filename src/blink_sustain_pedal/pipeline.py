"""The hot loop: latest frame → blendshapes → gesture → MIDI.

Blocks until the capture thread posts a fresh frame (no polling gap), then runs
inference on it immediately. Tracks per-stage timing so `--profile` / `calibrate`
can show whether you are camera-bound (frame age dominates) or compute-bound
(inference time dominates) — you almost always are the former.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable

from .camera import CameraStream
from .gestures import GestureDetector
from .landmarker import FaceBlendshapes
from .midi import SustainOutput

_READOUT_HZ = 20.0

# Action log. Emits nothing unless a handler is attached (the `run` command does
# this); INFO messages with no handler are dropped, so `calibrate` stays quiet.
logger = logging.getLogger(__name__)


def run(
    camera: CameraStream,
    landmarker: FaceBlendshapes,
    detector: GestureDetector,
    midi: SustainOutput | None,
    *,
    calibrate: bool = False,
    profile: bool = False,
    stop: Callable[[], bool] | None = None,
) -> None:
    start = time.monotonic()
    last_seq = 0
    last_ts = -1.0
    last_print = 0.0

    ema_infer_ms = 0.0     # smoothed inference time
    ema_age_ms = 0.0       # smoothed capture→process latency
    fps = 0.0
    fps_count = 0
    fps_since = start

    had_face: bool | None = None   # None until the first frame is classified
    down_since = 0.0               # monotonic time the pedal last went down

    try:
        while stop is None or not stop():
            seq, frame, stamp = camera.read_new(last_seq, timeout=0.1)
            if frame is None:
                continue  # timed out waiting for a frame; re-check stop()
            last_seq = seq

            proc_start = time.monotonic()
            frame_age_ms = (proc_start - stamp) * 1000.0

            # VIDEO mode needs strictly increasing timestamps; base them on capture time.
            ts = (stamp - start) * 1000.0
            if ts <= last_ts:
                ts = last_ts + 1.0
            last_ts = ts

            t0 = time.perf_counter()
            blendshapes = landmarker.detect(frame, ts)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            face = bool(blendshapes)
            if had_face is None:
                had_face = face
            elif face != had_face:
                logger.info("face %s", "reacquired" if face else "lost")
                had_face = face

            reading = detector.update(blendshapes, ts)
            if reading.changed:
                if midi is not None:
                    midi.set(reading.down)  # send first — logging must not add latency
                if reading.down:
                    down_since = proc_start
                    eye = "left" if reading.left >= reading.right else "right"
                    logger.info(
                        "pedal DOWN   eye=%-5s wink=%.2f other=%.2f",
                        eye, reading.wink_score, reading.other_score,
                    )
                else:
                    held_ms = (proc_start - down_since) * 1000.0 if down_since else 0.0
                    logger.info("pedal UP     held=%.0fms", held_ms)

            # --- timing stats ---
            ema_infer_ms = infer_ms if ema_infer_ms == 0.0 else 0.9 * ema_infer_ms + 0.1 * infer_ms
            ema_age_ms = frame_age_ms if ema_age_ms == 0.0 else 0.9 * ema_age_ms + 0.1 * frame_age_ms
            fps_count += 1
            now = time.monotonic()
            if now - fps_since >= 0.5:
                fps = fps_count / (now - fps_since)
                fps_count = 0
                fps_since = now

            if (calibrate or profile) and now - last_print >= 1.0 / _READOUT_HZ:
                _print_readout(
                    reading,
                    no_face=not blendshapes,
                    fps=fps,
                    infer_ms=ema_infer_ms,
                    age_ms=ema_age_ms,
                    show_scores=calibrate,
                )
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        if calibrate or profile:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _print_readout(reading, *, no_face, fps, infer_ms, age_ms, show_scores) -> None:
    parts = []
    if show_scores:
        state = "DOWN" if reading.down else "up  "
        parts.append(
            f"L {reading.left:0.2f}  R {reading.right:0.2f}  "
            f"| wink {reading.wink_score:0.2f}  other {reading.other_score:0.2f}  "
            f"| pedal {state}"
        )
    parts.append(f"| {fps:4.0f} fps  infer {infer_ms:4.1f}ms  frame-age {age_ms:4.1f}ms")
    if no_face:
        parts.append(" (no face)")
    sys.stdout.write("\r" + "  ".join(parts) + "   ")
    sys.stdout.flush()
