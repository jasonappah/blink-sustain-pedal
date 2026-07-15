"""Threaded webcam capture that always hands out the *latest* frame.

Latency is dominated by the camera, not inference, so the one thing that matters
here is never letting frames queue up: a dedicated thread grabs frames as fast as
the device delivers them and overwrites a single slot. Readers get the freshest
frame available and stale frames are simply dropped. A monotonically increasing
``seq`` lets the pipeline skip re-running inference on a frame it already processed.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2

_MACOS_PERMISSION_HINT = (
    "On macOS this usually means the terminal app hosting this process has not "
    "been granted camera access.\n"
    "  1. System Settings → Privacy & Security → Camera → enable your terminal.\n"
    "  2. If it isn't listed (macOS never prompted), reset and retry:\n"
    "       tccutil reset Camera <your-terminal-bundle-id>\n"
    "     e.g. for Warp:  tccutil reset Camera dev.warp.Warp-Stable\n"
    "     then fully quit and reopen the terminal and run this again."
)


def _camera_error(prefix: str) -> str:
    if sys.platform == "darwin":
        return f"{prefix}\n{_MACOS_PERMISSION_HINT}"
    return prefix


class CameraStream:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 60) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._seq = 0
        self._stamp = 0.0  # time.monotonic() when the current frame was captured
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "CameraStream":
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                _camera_error(
                    f"Could not open camera index {self.index}. "
                    "Try `blink-sustain-pedal list-cameras`."
                )
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Ask the driver to keep the smallest possible buffer (best-effort; some
        # macOS AVFoundation backends ignore this — the capture thread is the real fix).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self._cap = cap

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self._cap = None
            raise RuntimeError(
                _camera_error(
                    f"Camera index {self.index} opened but returned no frames."
                )
            )
        with self._cond:
            self._frame = frame
            self._seq = 1
            self._stamp = time.monotonic()

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()  # blocks until the driver delivers a frame
            stamp = time.monotonic()
            if not ok or frame is None:
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._stamp = stamp
                self._cond.notify_all()

    def read(self) -> tuple[int, object | None]:
        """Return ``(seq, frame_copy)``. ``seq`` is 0 and frame ``None`` before first frame."""
        with self._cond:
            if self._frame is None:
                return 0, None
            return self._seq, self._frame.copy()

    def read_new(self, last_seq: int, timeout: float = 0.1) -> tuple[int, object | None, float]:
        """Block until a frame newer than ``last_seq`` arrives (or ``timeout`` s).

        Returns ``(seq, frame_copy, capture_stamp)``. On timeout, frame is ``None``.
        This is the low-latency path: the consumer wakes the instant the capture
        thread posts a frame, with no polling gap.
        """
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._frame is None or self._seq == last_seq:
                return last_seq, None, 0.0
            return self._seq, self._frame.copy(), self._stamp

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def list_cameras(max_index: int = 6) -> list[dict]:
    """Best-effort probe of camera indices. macOS/OpenCV can't reliably name
    devices, so we report which indices open and the resolution they return."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    found: list[dict] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    found.append({"index": i, "width": w, "height": h})
        finally:
            cap.release()
    return found
