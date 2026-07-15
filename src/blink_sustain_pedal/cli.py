"""Typer CLI: run, calibrate, list-ports, list-cameras."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from .gestures import Eye, WinkDetector

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Wink your eye, sustain your notes — a webcam-driven MIDI sustain pedal.",
)

# --- shared option annotations -------------------------------------------------
CameraOpt = Annotated[int, typer.Option("--camera", "-c", help="Camera index.")]
WidthOpt = Annotated[int, typer.Option("--width", help="Capture width (px).")]
HeightOpt = Annotated[int, typer.Option("--height", help="Capture height (px).")]
FpsOpt = Annotated[int, typer.Option("--fps", help="Requested capture frame rate.")]
EyeOpt = Annotated[Eye, typer.Option("--eye", help="Which eye triggers the pedal.")]
OnOpt = Annotated[float, typer.Option("--on-threshold", help="Winking-eye score to engage.")]
OffOpt = Annotated[float, typer.Option("--off-threshold", help="Winking-eye score to release.")]
OpenOpt = Annotated[float, typer.Option("--open-threshold", help="Max score of the *other* eye to still count as a wink.")]
ReleaseOpt = Annotated[float, typer.Option("--release-ms", help="Hold after the eye reopens before releasing (ms).")]
GpuOpt = Annotated[bool, typer.Option("--gpu", help="Try the Metal GPU delegate (CPU is default and usually faster here).")]


def _build_detector(eye, on_t, off_t, open_t, release_ms) -> WinkDetector:
    try:
        return WinkDetector(
            eye=eye,
            on_threshold=on_t,
            off_threshold=off_t,
            open_threshold=open_t,
            release_ms=release_ms,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


@app.command()
def run(
    port: Annotated[Optional[str], typer.Option("--port", "-p", help="MIDI output name (case-insensitive substring). Defaults to the first port.")] = None,
    midi_cc: Annotated[int, typer.Option("--midi-cc", help="MIDI CC number (64 = sustain).")] = 64,
    midi_channel: Annotated[int, typer.Option("--midi-channel", help="MIDI channel 1..16.")] = 1,
    camera: CameraOpt = 0,
    width: WidthOpt = 640,
    height: HeightOpt = 480,
    fps: FpsOpt = 60,
    eye: EyeOpt = Eye.EITHER,
    on_threshold: OnOpt = 0.5,
    off_threshold: OffOpt = 0.3,
    open_threshold: OpenOpt = 0.4,
    release_ms: ReleaseOpt = 60.0,
    gpu: GpuOpt = False,
) -> None:
    """Detect winks and send MIDI sustain to a port (momentary: held wink = pedal down)."""
    # Imports deferred so `list-ports` / `--help` stay fast and don't need a camera.
    from . import midi as midi_mod
    from .camera import CameraStream
    from .landmarker import FaceBlendshapes
    from .pipeline import run as run_pipeline

    detector = _build_detector(eye, on_threshold, off_threshold, open_threshold, release_ms)
    try:
        port_name = midi_mod.find_port(port)
    except RuntimeError as exc:
        raise typer.Exit(code=_fail(str(exc)))

    landmarker = FaceBlendshapes(use_gpu=gpu)
    cam = CameraStream(index=camera, width=width, height=height, fps=fps)
    midi_out = None
    try:
        cam.start()
        midi_out = midi_mod.SustainOutput(port_name, cc=midi_cc, channel=midi_channel).open()
        typer.echo(f"MIDI → {port_name}   (CC{midi_cc}, channel {midi_channel})")
        typer.echo(f"Eye: {eye.value}.  Wink to sustain — Ctrl-C to quit.")
        run_pipeline(cam, landmarker, detector, midi_out, calibrate=False)
    finally:
        if midi_out is not None:
            midi_out.close()
        cam.stop()
        landmarker.close()


@app.command()
def calibrate(
    camera: CameraOpt = 0,
    width: WidthOpt = 640,
    height: HeightOpt = 480,
    fps: FpsOpt = 60,
    eye: EyeOpt = Eye.EITHER,
    on_threshold: OnOpt = 0.5,
    off_threshold: OffOpt = 0.3,
    open_threshold: OpenOpt = 0.4,
    release_ms: ReleaseOpt = 60.0,
    gpu: GpuOpt = False,
) -> None:
    """Print live per-eye blink scores and pedal state to the terminal for tuning (no MIDI)."""
    from .camera import CameraStream
    from .landmarker import FaceBlendshapes
    from .pipeline import run as run_pipeline

    detector = _build_detector(eye, on_threshold, off_threshold, open_threshold, release_ms)
    landmarker = FaceBlendshapes(use_gpu=gpu)
    cam = CameraStream(index=camera, width=width, height=height, fps=fps)
    try:
        cam.start()
        typer.echo(
            f"Calibrating (eye={eye.value}, on={on_threshold}, off={off_threshold}, "
            f"open={open_threshold}). Wink each eye, then blink both. Ctrl-C to quit."
        )
        run_pipeline(cam, landmarker, detector, None, calibrate=True)
    finally:
        cam.stop()
        landmarker.close()


@app.command("list-ports")
def list_ports() -> None:
    """List available MIDI output ports."""
    from . import midi as midi_mod

    ports = midi_mod.list_output_ports()
    if not ports:
        typer.echo("No MIDI output ports found.")
        raise typer.Exit(code=1)
    typer.echo("MIDI output ports:")
    for name in ports:
        typer.echo(f"  • {name}")


@app.command("list-cameras")
def list_cameras() -> None:
    """Probe and list available camera indices."""
    from .camera import list_cameras as probe

    cams = probe()
    if not cams:
        typer.echo("No cameras found.")
        raise typer.Exit(code=1)
    typer.echo("Cameras:")
    for cam in cams:
        typer.echo(f"  • index {cam['index']}  ({cam['width']}x{cam['height']})")


def _fail(message: str) -> int:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return 1


if __name__ == "__main__":
    app()
