"""MIDI sustain output over an existing system port (mido + python-rtmidi).

Sends a Control Change (CC64 = sustain by default): value 127 = pedal down,
0 = pedal up. Always releases the pedal on close so a note can never hang in the
DAW if the app exits mid-wink.
"""

from __future__ import annotations

import mido


def list_output_ports() -> list[str]:
    return list(mido.get_output_names())


def find_port(substring: str | None) -> str:
    """Resolve a MIDI output name: first port if ``substring`` is None, else the
    first case-insensitive substring match."""
    ports = mido.get_output_names()
    if not ports:
        raise RuntimeError(
            "No MIDI output ports found. On macOS, enable the IAC Driver in "
            "Audio MIDI Setup (or connect a MIDI device) first."
        )
    if substring is None:
        return ports[0]
    matches = [p for p in ports if substring.lower() in p.lower()]
    if not matches:
        raise RuntimeError(
            f"No MIDI output matching {substring!r}. Available: {ports}"
        )
    return matches[0]


class SustainOutput:
    def __init__(self, port_name: str, cc: int = 64, channel: int = 1) -> None:
        if not 0 <= cc <= 127:
            raise ValueError("cc must be within 0..127")
        if not 1 <= channel <= 16:
            raise ValueError("channel must be within 1..16")
        self.port_name = port_name
        self.cc = cc
        self.channel = channel - 1  # mido uses 0-based channels
        self._port: mido.ports.BaseOutput | None = None
        self._down = False

    def open(self) -> "SustainOutput":
        self._port = mido.open_output(self.port_name)
        return self

    def set(self, down: bool) -> None:
        if self._port is None:
            raise RuntimeError("MIDI port is not open")
        self._port.send(
            mido.Message(
                "control_change",
                channel=self.channel,
                control=self.cc,
                value=127 if down else 0,
            )
        )
        self._down = down

    def close(self) -> None:
        if self._port is None:
            return
        try:
            if self._down:
                self.set(False)  # never leave the pedal stuck down
        finally:
            self._port.close()
            self._port = None

    def __enter__(self) -> "SustainOutput":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()
