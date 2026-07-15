"""MediaPipe Face Landmarker wrapper that yields per-frame blendshape scores.

Runs the Tasks API in VIDEO mode (synchronous, deterministic — we call it on the
latest frame and get the result back immediately). The only outputs we need are the
52 blendshape coefficients, of which ``eyeBlinkLeft`` / ``eyeBlinkRight`` drive the
pedal. The ~3.7 MB model bundle is downloaded once to ~/.cache on first run.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# float16 bundle includes the face-blendshapes head.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "blink-sustain-pedal" / "face_landmarker.task"


def ensure_model(path: Path | None = None) -> Path:
    """Return a path to the model bundle, downloading it once if absent."""
    dest = path or DEFAULT_MODEL_PATH
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading face landmarker model (~4 MB)…", file=sys.stderr, flush=True)
    tmp = dest.with_suffix(".download")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


class FaceBlendshapes:
    def __init__(self, model_path: str | None = None, use_gpu: bool = False) -> None:
        model = ensure_model(Path(model_path) if model_path else None)
        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if use_gpu
            else mp_python.BaseOptions.Delegate.CPU
        )
        base = mp_python.BaseOptions(model_asset_path=str(model), delegate=delegate)
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms: float) -> dict[str, float]:
        """Return ``{blendshape_name: score}`` for the first detected face (empty if none)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.face_blendshapes:
            return {}
        return {c.category_name: c.score for c in result.face_blendshapes[0]}

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceBlendshapes":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
