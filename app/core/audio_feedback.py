"""
audio_feedback.py
==================

Folder location: PhysioAI/app/core/audio_feedback.py

Purpose
-------
Provides offline, non-blocking audio feedback (speech + short sound
effects) for physiotherapy/rehabilitation exercise sessions, so
patients get real-time cues without the video/analysis loop ever
stalling on audio playback.

Design notes
------------
- NON-BLOCKING: speech is generated on a dedicated background daemon
  thread reading from a thread-safe priority queue. `speak()` /
  `queue_feedback()` only enqueue a message and return immediately;
  the caller's video loop never waits on pyttsx3.
- OFFLINE ONLY: uses pyttsx3 for text-to-speech (no network calls) and
  pygame.mixer for short .wav sound effects (success/warning/error).
- GRACEFUL DEGRADATION: if pyttsx3 or pygame is not installed, or an
  engine fails to initialize, the corresponding feature is disabled
  with a logged warning instead of raising and crashing the app.
- NO GLOBAL STATE: everything lives on the AudioFeedback instance
  (queue, locks, last-spoken timestamps, engine handle).
- ASSET PATHS: sound-effect paths are resolved with pathlib relative
  to this file's location (PhysioAI/assets/sounds/...), not relative
  to the process's current working directory, so playback works the
  same whether the app is launched from the project root, from
  run.py in a different folder, or from a test runner.

Author: Ramya (Core module)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None  # type: ignore[assignment]

try:
    import pygame
except ImportError:  # pragma: no cover - optional dependency
    pygame = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SPEAKING_RATE: int = 175         # words per minute (pyttsx3 default ~200)
DEFAULT_VOLUME: float = 1.0              # 0.0 - 1.0
DEFAULT_REPEAT_COOLDOWN_SECONDS: float = 3.0  # suppress identical message repeats

# Resolve assets relative to this file's location rather than the
# current working directory:
#   PhysioAI/
#   ├── app/core/audio_feedback.py   <- this file
#   └── assets/sounds/               <- sound assets live here
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOUND_ASSET_DIR: Path = _PROJECT_ROOT / "assets" / "sounds"
SUCCESS_SOUND_FILE: Path = SOUND_ASSET_DIR / "success.wav"
WARNING_SOUND_FILE: Path = SOUND_ASSET_DIR / "warning.wav"
ERROR_SOUND_FILE: Path = SOUND_ASSET_DIR / "error.wav"

QUEUE_POLL_TIMEOUT_SECONDS: float = 0.5   # how often the worker checks for shutdown
SHUTDOWN_JOIN_TIMEOUT_SECONDS: float = 2.0


class FeedbackPriority(IntEnum):
    """Higher-priority messages are spoken before lower-priority ones."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


# ---------------------------------------------------------------------------
# Internal message container
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _FeedbackMessage:
    """
    Queue item. `sort_key` is negative priority so that PriorityQueue's
    natural min-heap ordering pops the HIGHEST priority first; all other
    fields are excluded from comparison so equal-priority items don't
    need a defined ordering between their text/timestamp.
    """
    sort_key: int
    text: str = field(compare=False)
    timestamp: float = field(compare=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class AudioFeedback:
    """
    Offline, thread-safe audio feedback engine combining text-to-speech
    and short sound effects.

    Typical usage:

        audio = AudioFeedback()
        audio.speak("Raise your hand higher")
        audio.play_success()
        ...
        audio.shutdown()   # call once, when the application exits
    """

    def __init__(
        self,
        speaking_rate: int = DEFAULT_SPEAKING_RATE,
        volume: float = DEFAULT_VOLUME,
        voice_enabled: bool = True,
        sound_effects_enabled: bool = True,
        repeat_cooldown_seconds: float = DEFAULT_REPEAT_COOLDOWN_SECONDS,
    ) -> None:
        """
        Args:
            speaking_rate: pyttsx3 speech rate in words per minute.
            volume: Speech volume, 0.0 (silent) to 1.0 (max).
            voice_enabled: Whether spoken feedback starts enabled.
            sound_effects_enabled: Whether sound effects start enabled.
            repeat_cooldown_seconds: Minimum seconds before the exact
                same message text can be spoken again.
        """
        self._voice_enabled: bool = voice_enabled
        self._sound_effects_enabled: bool = sound_effects_enabled
        self._repeat_cooldown_seconds: float = repeat_cooldown_seconds

        self._message_queue: "queue.PriorityQueue[Optional[_FeedbackMessage]]" = (
            queue.PriorityQueue()
        )
        self._last_spoken: Dict[str, float] = {}
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._engine = self._init_tts_engine(speaking_rate, volume)
        self._mixer_ready = self._init_sound_mixer()

        self._worker_thread = threading.Thread(
            target=self._process_queue, daemon=True, name="AudioFeedbackWorker"
        )
        self._worker_thread.start()
        logger.info(
            "AudioFeedback initialized (voice=%s, sfx=%s).",
            self._engine is not None, self._mixer_ready,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_tts_engine(self, rate: int, volume: float):
        """Initialize pyttsx3 offline TTS engine, or return None on failure."""
        if pyttsx3 is None:
            logger.warning("pyttsx3 is not installed; voice feedback disabled.")
            return None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", max(0.0, min(1.0, volume)))
            return engine
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize pyttsx3 engine: %s", exc)
            return None

    def _init_sound_mixer(self) -> bool:
        """Initialize pygame's mixer for sound effects. Returns True on success."""
        if pygame is None:
            logger.warning("pygame is not installed; sound effects disabled.")
            return False
        try:
            pygame.mixer.init()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize pygame mixer: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API -- speech
    # ------------------------------------------------------------------
    def speak(self, text: str, priority: FeedbackPriority = FeedbackPriority.NORMAL) -> None:
        """
        Enqueue a spoken feedback message. Returns immediately; the
        message is spoken asynchronously on the worker thread.

        Args:
            text: The sentence to speak.
            priority: Playback priority; higher-priority messages jump
                ahead of lower-priority ones still waiting in the queue.
        """
        if not self._voice_enabled or self._engine is None:
            return
        if not text or not text.strip():
            return
        if self._is_duplicate(text):
            logger.debug("Suppressed repeated message within cooldown: %r", text)
            return
        self._message_queue.put(
            _FeedbackMessage(sort_key=-int(priority), text=text, timestamp=time.time())
        )

    def queue_feedback(self, text: str, priority: FeedbackPriority = FeedbackPriority.NORMAL) -> None:
        """
        Public alias for `speak()`. Provided as a distinct, explicit
        name for call sites that are conceptually "queuing feedback for
        later" (e.g. batched end-of-repetition feedback) rather than an
        immediate cue -- behavior is identical to `speak()`.

        Args:
            text: The sentence to queue for speaking.
            priority: Playback priority.
        """
        self.speak(text, priority)

    # ------------------------------------------------------------------
    # Public API -- sound effects
    # ------------------------------------------------------------------
    def play_success(self) -> None:
        """Play the short 'success' sound effect (e.g. repetition completed correctly)."""
        self._play_sound_effect(SUCCESS_SOUND_FILE)

    def play_warning(self) -> None:
        """Play the short 'warning' sound effect (e.g. movement quality dropping)."""
        self._play_sound_effect(WARNING_SOUND_FILE)

    def play_error(self) -> None:
        """Play the short 'error' sound effect (e.g. tracking lost, invalid pose)."""
        self._play_sound_effect(ERROR_SOUND_FILE)

    # ------------------------------------------------------------------
    # Public API -- configuration
    # ------------------------------------------------------------------
    def set_speaking_rate(self, rate: int) -> None:
        """
        Adjust the text-to-speech rate.

        Args:
            rate: Words per minute (pyttsx3 typical range ~100-300).
        """
        if self._engine is not None:
            try:
                self._engine.setProperty("rate", rate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set speaking rate: %s", exc)

    def set_volume(self, volume: float) -> None:
        """
        Adjust the text-to-speech volume.

        Args:
            volume: Value from 0.0 (silent) to 1.0 (maximum), clamped.
        """
        if self._engine is not None:
            try:
                self._engine.setProperty("volume", max(0.0, min(1.0, volume)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set volume: %s", exc)

    def enable_voice(self, enabled: bool) -> None:
        """Enable or disable spoken feedback without tearing down the engine."""
        with self._state_lock:
            self._voice_enabled = enabled

    def enable_sound_effects(self, enabled: bool) -> None:
        """Enable or disable short sound-effect playback."""
        with self._state_lock:
            self._sound_effects_enabled = enabled

    # ------------------------------------------------------------------
    # Public API -- lifecycle
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        Immediately stop any current/pending speech: clears the queue
        and interrupts the TTS engine if it is mid-utterance. The
        worker thread keeps running and is ready for new messages.
        """
        with self._state_lock:
            while True:
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    break
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping TTS engine: %s", exc)

    def shutdown(self) -> None:
        """
        Gracefully stop the worker thread and release audio resources.
        Must be called once when the application is closing (e.g. from
        run.py's cleanup path) to avoid a dangling background thread.
        """
        self.stop()
        self._stop_event.set()
        # Sentinel value to unblock a worker thread that is currently
        # waiting on queue.get().
        self._message_queue.put(_FeedbackMessage(sort_key=999, text="", timestamp=0.0))
        self._worker_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)

        if pygame is not None and self._mixer_ready:
            try:
                pygame.mixer.quit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error shutting down pygame mixer: %s", exc)

        logger.info("AudioFeedback shut down cleanly.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_duplicate(self, text: str) -> bool:
        """
        Check whether `text` was spoken within the repeat cooldown
        window, and record this attempt's timestamp either way.

        Args:
            text: Candidate message text.

        Returns:
            True if this exact message should be suppressed as a
            duplicate; False if it is allowed through.
        """
        now = time.time()
        with self._state_lock:
            last_time = self._last_spoken.get(text)
            if last_time is not None and (now - last_time) < self._repeat_cooldown_seconds:
                return True
            self._last_spoken[text] = now
        return False

    def _play_sound_effect(self, filepath: Path) -> None:
        """
        Play a short .wav sound effect via pygame, if enabled and
        available. Missing files or playback errors are logged, never
        raised, so a missing asset can't crash the analysis loop.

        Args:
            filepath: Path to the .wav file to play.
        """
        with self._state_lock:
            enabled = self._sound_effects_enabled
        if not enabled or not self._mixer_ready:
            return
        if not filepath.exists():
            logger.warning("Sound effect file not found: %s", filepath)
            return
        try:
            sound = pygame.mixer.Sound(str(filepath))
            sound.play()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to play sound effect '%s': %s", filepath, exc)

    def _process_queue(self) -> None:
        """
        Worker-thread loop: pulls messages off the priority queue and
        speaks them sequentially via pyttsx3. Runs until `shutdown()`
        signals the stop event. This is the ONLY place `runAndWait()`
        is called, keeping all blocking TTS calls off the main thread.
        """
        while not self._stop_event.is_set():
            try:
                message = self._message_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break
            if not message.text:
                # Shutdown sentinel (empty text) -- nothing to speak.
                continue
            if self._engine is None:
                continue
            try:
                self._engine.say(message.text)
                self._engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                logger.error("TTS playback failed for %r: %s", message.text, exc)