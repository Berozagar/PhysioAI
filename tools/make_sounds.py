"""
make_sounds.py
==============

Generates the three short sound effects AudioFeedback expects:

    assets/sounds/success.wav
    assets/sounds/warning.wav
    assets/sounds/error.wav

The repository ships no audio assets, so without these
AudioFeedback.play_success() / play_warning() / play_error() silently
do nothing (they log "file not found" and return). Running this script
once makes the sound-effect path functional.

Uses only the Python standard library -- no extra dependencies.

Run from the project root:

    python tools/make_sounds.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Sequence, Tuple

SAMPLE_RATE: int = 44_100
AMPLITUDE: float = 0.35          # keep well below clipping
FADE_SECONDS: float = 0.008      # short fade to avoid click artifacts

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOUND_DIR = _PROJECT_ROOT / "assets" / "sounds"

# (frequency_hz, duration_seconds) sequences played back to back.
# success: rising major arpeggio  -> "you did it"
# warning: two flat mid tones     -> "check your form"
# error:   descending low tones   -> "something is wrong"
TONE_SEQUENCES: dict[str, Sequence[Tuple[float, float]]] = {
    "success": ((523.25, 0.09), (659.25, 0.09), (783.99, 0.20)),
    "warning": ((440.00, 0.11), (0.0, 0.05), (440.00, 0.16)),
    "error": ((311.13, 0.16), (233.08, 0.26)),
}


def _envelope(index: int, total: int) -> float:
    """Linear fade-in/fade-out multiplier for sample `index` of `total`."""
    fade_samples = max(1, int(FADE_SECONDS * SAMPLE_RATE))
    if index < fade_samples:
        return index / fade_samples
    remaining = total - index
    if remaining < fade_samples:
        return remaining / fade_samples
    return 1.0


def _render_tone(frequency: float, duration: float) -> list[float]:
    """Render one sine tone (or silence when frequency is 0) as floats in [-1, 1]."""
    sample_count = int(duration * SAMPLE_RATE)
    if frequency <= 0.0:
        return [0.0] * sample_count

    step = 2.0 * math.pi * frequency / SAMPLE_RATE
    return [
        AMPLITUDE * math.sin(step * i) * _envelope(i, sample_count)
        for i in range(sample_count)
    ]


def _write_wav(path: Path, samples: Sequence[float]) -> None:
    """Write float samples to a 16-bit mono WAV file."""
    frames = b"".join(
        struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)) for sample in samples
    )
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(frames)


def main() -> None:
    SOUND_DIR.mkdir(parents=True, exist_ok=True)

    for name, tones in TONE_SEQUENCES.items():
        samples: list[float] = []
        for frequency, duration in tones:
            samples.extend(_render_tone(frequency, duration))

        target = SOUND_DIR / f"{name}.wav"
        _write_wav(target, samples)
        print(f"wrote {target}  ({len(samples) / SAMPLE_RATE:.2f}s, {target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
