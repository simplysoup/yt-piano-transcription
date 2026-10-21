import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pretty_midi

from scoreify import (
    detect_ornaments,
    fit_beat_phase,
    nearest_grid_beat,
    parse_meter,
    parse_swing_ratio,
    write_midi_preserving_times,
)


class ScoreifyTests(unittest.TestCase):
    def test_parse_meter_and_swing(self):
        self.assertEqual(parse_meter("3/4"), (3, 4))
        self.assertEqual(parse_swing_ratio("2:1"), 2.0)

    def test_swung_eighth_maps_to_even_notated_eighth(self):
        beat = 60 / 158
        played_offbeat = (2 / 3) * beat
        result = nearest_grid_beat(played_offbeat, beat, 0.0, 2.0, 4)
        self.assertTrue(math.isclose(result, 0.5, abs_tol=1e-9))

    def test_phase_fit_recovers_synthetic_swing_grid(self):
        beat = 60 / 158
        true_phase = 0.137
        swing = 2 / 3
        onsets = []
        for i in range(24):
            onsets.extend((true_phase + i * beat, true_phase + (i + swing) * beat))
        fitted = fit_beat_phase(onsets, beat, 2.0, 4)
        delta = abs(fitted - true_phase)
        delta = min(delta, beat - delta)
        self.assertLess(delta, beat / 100)

    def test_preserve_writer_keeps_absolute_note_times(self):
        performance = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0, name="Piano")
        instrument.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=1.721, end=1.903))
        performance.instruments.append(instrument)
        item = {"id": 0, "instrument": 0, "note": instrument.notes[0]}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "preserved.mid"
            write_midi_preserving_times(performance, [item], path, (4, 4))
            result = pretty_midi.PrettyMIDI(str(path))
            note = result.instruments[0].notes[0]
            self.assertAlmostEqual(note.start, 1.721, places=2)
            self.assertAlmostEqual(note.end, 1.903, places=2)

    def test_flags_short_stepwise_run_before_long_note(self):
        starts = [0.0, 0.08, 0.16, 0.25]
        pitches = [64, 66, 67, 69]
        durations = [0.06, 0.06, 0.06, 0.35]
        items = []
        for idx, (start, pitch, length) in enumerate(zip(starts, pitches, durations)):
            items.append({
                "id": idx,
                "note": pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + length),
            })
        groups = detect_ornaments(items, beat_seconds=0.5, split_pitch=60)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["note_ids"], [0, 1, 2])
        self.assertEqual(groups[0]["principal_note_id"], 3)


if __name__ == "__main__":
    unittest.main()
