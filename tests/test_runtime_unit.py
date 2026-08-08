"""
Tests for the runtime pieces: the FPS counter, the audio feedback queue
logic, and run.py's entrypoint driven end to end via --self-test.

Audio tests never actually speak: the engine is constructed once and left
with voice and sound effects disabled.
"""

import time

import pytest

import run as app_run
from app.core.audio_feedback import AudioFeedback, FeedbackPriority, _FeedbackMessage
from app.utils.fps import FPSCounter


class TestFPSCounter:
    def test_starts_at_zero(self):
        assert FPSCounter().fps == 0.0

    def test_update_returns_a_positive_rate(self):
        counter = FPSCounter()
        time.sleep(0.02)
        assert counter.update() > 0.0

    def test_rate_reflects_the_elapsed_interval(self):
        counter = FPSCounter()
        time.sleep(0.1)
        # ~10 FPS; generous bounds so a loaded machine does not flake.
        assert 3.0 < counter.update() < 40.0

    def test_result_is_rounded_to_two_decimals(self):
        counter = FPSCounter()
        time.sleep(0.02)
        value = counter.update()
        assert value == round(value, 2)

    def test_repeated_updates_keep_returning_values(self):
        counter = FPSCounter()
        for _ in range(3):
            time.sleep(0.01)
            assert counter.update() > 0.0

    def test_update_advances_the_reference_time(self):
        counter = FPSCounter()
        first = counter.previous_time
        time.sleep(0.01)
        counter.update()
        assert counter.previous_time > first


class TestFeedbackMessageOrdering:
    def test_higher_priority_sorts_first(self):
        """sort_key is negated priority so a min-heap pops HIGH first."""
        high = _FeedbackMessage(sort_key=-int(FeedbackPriority.HIGH), text="a", timestamp=0.0)
        low = _FeedbackMessage(sort_key=-int(FeedbackPriority.LOW), text="b", timestamp=0.0)
        assert high < low

    def test_equal_priority_messages_compare_equal(self):
        first = _FeedbackMessage(sort_key=-1, text="a", timestamp=1.0)
        second = _FeedbackMessage(sort_key=-1, text="z", timestamp=99.0)
        assert not (first < second) and not (second < first)

    def test_priority_levels_are_ordered(self):
        assert FeedbackPriority.LOW < FeedbackPriority.NORMAL < FeedbackPriority.HIGH


class TestAudioFeedback:
    @pytest.fixture(scope="class")
    def audio(self):
        engine = AudioFeedback(voice_enabled=False, sound_effects_enabled=False)
        yield engine
        engine.shutdown()

    def test_worker_thread_is_running(self, audio):
        assert audio._worker_thread.is_alive()

    def test_repeat_within_cooldown_is_suppressed(self, audio):
        text = "unique message for cooldown test"
        assert audio._is_duplicate(text) is False   # first time through
        assert audio._is_duplicate(text) is True    # immediately again

    def test_distinct_messages_are_not_suppressed(self, audio):
        assert audio._is_duplicate("message one") is False
        assert audio._is_duplicate("message two") is False

    def test_speak_is_ignored_while_voice_disabled(self, audio):
        audio.speak("this should not be queued")
        assert audio._message_queue.empty()

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_text_is_never_queued(self, audio, blank):
        audio.enable_voice(True)
        try:
            audio.speak(blank)
            assert audio._message_queue.empty()
        finally:
            audio.enable_voice(False)

    def test_missing_sound_file_is_logged_not_raised(self, audio, tmp_path):
        audio._play_sound_effect(tmp_path / "absent.wav")  # must not raise

    def test_enable_flags_round_trip(self, audio):
        audio.enable_sound_effects(True)
        assert audio._sound_effects_enabled is True
        audio.enable_sound_effects(False)
        assert audio._sound_effects_enabled is False

    def test_setters_tolerate_out_of_range_values(self, audio):
        audio.set_volume(5.0)     # clamped, must not raise
        audio.set_speaking_rate(0)

    def test_stop_drains_the_queue(self, audio):
        audio._message_queue.put(_FeedbackMessage(sort_key=0, text="x", timestamp=0.0))
        audio.stop()
        assert audio._message_queue.empty()


class TestEntrypointSelfTest:
    @pytest.mark.parametrize("exercise", ["shoulder_raise", "squat", "knee_bend"])
    def test_self_test_passes_for_every_exercise(self, exercise, capsys):
        code = app_run.main(
            ["--self-test", "--exercise", exercise, "--no-audio", "--self-test-reps", "2"]
        )
        assert code == 0
        assert "SELF-TEST PASSED" in capsys.readouterr().out

    def test_self_test_counts_the_requested_reps(self, capsys):
        app_run.main(["--self-test", "--no-audio", "--self-test-reps", "3"])
        assert "reps=3" in capsys.readouterr().out

    def test_self_test_works_without_smoothing(self, capsys):
        code = app_run.main(["--self-test", "--no-audio", "--smoothing", "0"])
        assert code == 0
        assert "SELF-TEST PASSED" in capsys.readouterr().out

    def test_saves_an_annotated_frame(self, tmp_path, capsys):
        target = tmp_path / "frame.png"
        app_run.main(
            ["--self-test", "--no-audio", "--self-test-reps", "1",
             "--save-frame", str(target)]
        )
        assert target.exists() and target.stat().st_size > 0

    def test_unwritable_save_path_warns_instead_of_claiming_success(
        self, tmp_path, capsys
    ):
        """Regression: imwrite failure used to print 'Saved ...' regardless."""
        target = tmp_path / "missing_dir" / "frame.png"
        app_run.main(
            ["--self-test", "--no-audio", "--self-test-reps", "1",
             "--save-frame", str(target)]
        )
        captured = capsys.readouterr()
        assert not target.exists()
        assert "could not write" in captured.err
        assert "Saved last annotated frame" not in captured.out

    def test_bad_video_source_exits_nonzero(self, capsys):
        code = app_run.main(["--source", "definitely_not_a_file.mp4", "--no-audio"])
        assert code == 1
        assert "could not open video source" in capsys.readouterr().err
