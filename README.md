# blink-sustain-pedal

🎹 eyelid closed, pedal down.

A CLI that watches your webcam, detects a **single-eye wink**, and sends **MIDI
sustain (CC64)** to an existing system MIDI port — so you can hold sustain in
Ableton (or any DAW) without a hardware pedal. Inspired by Planning Center Music
Stand's blink-to-turn-page feature.

- **Momentary**, like a real pedal: sustain is on while the wink is held.
- **Wink, not blink:** a wink closes one eye while the other stays open; normal
  two-eye blinks are ignored, so involuntary blinking never triggers the pedal.
- **Low latency:** a latest-frame-wins capture thread + MediaPipe Face Landmarker.
  End-to-end is ~25–50 ms, dominated by the camera, not inference.

## Requirements

- macOS on Apple Silicon (developed there; Linux likely works via V4L cameras).
- Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/).
- A MIDI output that already exists on your system. On macOS, enable the **IAC
  Driver** in *Audio MIDI Setup → Window → Show MIDI Studio → IAC Driver → "Device
  is online"* to get a loopback bus (e.g. `IAC Driver Bus 1`).

## Install

```bash
uv sync
```

The ~4 MB face-landmarker model downloads automatically on first run to
`~/.cache/blink-sustain-pedal/`.

## Usage

```bash
# What can I talk to?
uv run blink-sustain-pedal list-ports
uv run blink-sustain-pedal list-cameras

# Tune first: prints live blink scores, no MIDI sent.
# Wink each eye, then blink both, and watch the numbers.
uv run blink-sustain-pedal calibrate

# Go. Sends CC64 to the first MIDI port matching "IAC".
uv run blink-sustain-pedal run --port IAC
```

Then in Ableton: enable the IAC bus as a MIDI input (Preferences → Link/MIDI), arm
an instrument track, hold a note, and wink-and-hold — the note sustains until you
open your eye.

### Handy options (`run` and `calibrate`)

| Option | Default | Meaning |
| --- | --- | --- |
| `--eye left\|right\|either` | `either` | Which eye triggers the pedal |
| `--port` | first port | MIDI output (case-insensitive substring match) |
| `--midi-cc` / `--midi-channel` | `64` / `1` | Which CC and channel to send |
| `--on-threshold` / `--off-threshold` | `0.5` / `0.3` | Engage / release scores (hysteresis) |
| `--open-threshold` | `0.4` | How open the *other* eye must be to count as a wink |
| `--release-ms` | `60` | Debounce hold after the eye reopens |
| `--camera` / `--width` / `--height` / `--fps` | `0` / `640` / `480` / `60` | Capture settings |
| `--gpu` | off | Try the Metal GPU delegate (CPU is default and usually faster here) |

Run `uv run blink-sustain-pedal run --help` for the full list.

## Lowering latency

Run with `--profile` (or use `calibrate`, which always shows it) to see where time
goes:

```bash
uv run blink-sustain-pedal run --port IAC --profile
#  → 60 fps  infer 6.2ms  frame-age 15.8ms
```

- **`frame-age` ≫ `infer`** (the usual case): you're **camera-bound**. Fixes, in order
  of impact:
  - **More light.** In dim rooms the camera lengthens exposure, which both blurs and
    *slows* frames. Good lighting shortens exposure and often restores full fps.
  - **Confirm real fps.** Many built-in cameras drop below their rated fps at higher
    resolution or in low light. A 60/120 fps external webcam — or an iPhone via
    Continuity Camera — delivers fresher frames. Lower `--width`/`--height` can unlock
    a higher `--fps`.
- **`infer` is high (>20ms):** try `--gpu` (Metal delegate — may or may not beat CPU;
  measure), and keep your face in frame so MediaPipe stays in its cheap tracking path.
- **Snappier feel, independent of the camera:**
  - Lower `--on-threshold` (e.g. `0.35`) so the pedal engages *earlier* in the eyelid's
    closing motion — trades a little false-trigger margin for lower attack latency.
  - Lower `--release-ms` (e.g. `25`) for a crisper pedal-up.
- **In Ableton:** the audio **buffer size** (Preferences → Audio) dominates *audible*
  output latency. That said, sustain (CC64) is timing-forgiving — a few ms early/late is
  inaudible — so chase single-digit milliseconds only if you actually feel it.

## macOS permissions

The first time you run `calibrate` or `run` from your terminal, macOS will prompt
to grant **camera access** to that terminal app. Approve it (System Settings →
Privacy & Security → Camera). Until you do, the camera opens with no frames.

## Notes & limitations

- **Neural Engine:** the Python MediaPipe API only exposes CPU/GPU delegates, so
  the ANE isn't used — and it wouldn't help, since the tiny model already runs
  faster than the camera delivers frames. CPU is the reliable default.
- **Tongue-out won't work** off-the-shelf: MediaPipe's blendshape model
  deliberately zeroes `tongueOut` (and `jawForward`, `cheekPuff`). Gestures that
  *do* work well if you want to extend this: `jawOpen`, `browInnerUp`,
  `mouthPucker`, `eyeSquint*`. The gesture layer (`gestures.py`) is a small
  pluggable `GestureDetector` protocol, so adding one is a few lines.

## Development

```bash
uv run pytest        # unit tests for the wink state machine (no camera/MIDI needed)
```

Module map (`src/blink_sustain_pedal/`): `camera.py` (threaded capture),
`landmarker.py` (MediaPipe wrapper + model download), `gestures.py` (pure
wink→pedal logic), `midi.py` (CC output), `pipeline.py` (hot loop), `cli.py` (Typer).
