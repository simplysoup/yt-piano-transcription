#!/usr/bin/env python3
"""Convert Transkun performance MIDI into beat-grid score variants."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi

from pipeline import make_musicxml, render_pdf


def parse_meter(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)/(\d+)", value.strip())
    if not match:
        raise ValueError("meter must look like 3/4 or 4/4")
    numerator, denominator = map(int, match.groups())
    if numerator < 1 or denominator not in (2, 4, 8, 16):
        raise ValueError("meter numerator must be positive; denominator must be 2, 4, 8, or 16")
    return numerator, denominator


def parse_swing_ratio(value: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", value)
    if not match:
        raise ValueError("swing ratio must be written as long:short, e.g. 2:1 or 3:2")
    long, short = map(float, match.groups())
    if long < short or short <= 0:
        raise ValueError("swing ratio must have long >= short > 0")
    return long / short


def estimate_bpm(wav_path: Path) -> float:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Install librosa or pass --bpm explicitly") from exc
    audio, sample_rate = librosa.load(str(wav_path), sr=22_050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate, units="time", trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    if not math.isfinite(bpm) or bpm <= 0:
        raise RuntimeError("beat tracker did not return a usable tempo; pass --bpm")
    return bpm


def cluster_onsets(notes: list[dict], tolerance: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for item in sorted(notes, key=lambda row: row["note"].start):
        start = item["note"].start
        if not clusters or start - clusters[-1][-1]["note"].start > tolerance:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def fit_beat_phase(onsets: list[float], beat_seconds: float, swing_ratio: float, grid_divisions: int) -> float:
    """Fit a beat phase to note onsets using the requested swung rhythmic grid."""
    if not onsets:
        return 0.0
    times = np.asarray(onsets, dtype=float)
    phases = np.linspace(0.0, beat_seconds, 512, endpoint=False)
    relative = (times[:, None] - phases[None, :]) / beat_seconds
    fractional = relative - np.floor(relative)
    swing_position = swing_ratio / (swing_ratio + 1.0)
    notation_positions = np.arange(grid_divisions, dtype=float) / grid_divisions
    performance_positions = notation_positions.copy()
    midpoint_idx = np.flatnonzero(np.isclose(notation_positions, 0.5))
    if len(midpoint_idx):
        performance_positions[midpoint_idx[0]] = swing_position

    raw = np.abs(fractional[:, :, None] - performance_positions[None, None, :])
    distances = np.minimum(raw, 1.0 - raw).min(axis=2)
    # Truncate outliers so ornaments and syncopations do not dominate phase fitting.
    costs = np.minimum(distances, 0.25).mean(axis=0)
    return float(phases[int(np.argmin(costs))])


def nearest_grid_beat(time_seconds: float, beat_seconds: float, phase: float,
                      swing_ratio: float, grid_divisions: int) -> float:
    relative = (time_seconds - phase) / beat_seconds
    lower = math.floor(relative)
    swing_position = swing_ratio / (swing_ratio + 1.0)
    notation_positions = [i / grid_divisions for i in range(grid_divisions)]
    performance_positions = list(notation_positions)
    if grid_divisions % 2 == 0:
        midpoint_idx = grid_divisions // 2
        performance_positions[midpoint_idx] = swing_position

    best_score = None
    best_beat = relative
    for beat_index in range(lower - 1, lower + 3):
        for notation_pos, performance_pos in zip(notation_positions, performance_positions):
            candidate_performance = beat_index + performance_pos
            deviation = abs(relative - candidate_performance)
            score = (deviation, abs(notation_pos - (relative - beat_index)))
            if best_score is None or score < best_score:
                best_score = score
                best_beat = beat_index + notation_pos
    return best_beat


def detect_ornaments(items: list[dict], beat_seconds: float, split_pitch: int = 60) -> list[dict]:
    """Conservatively flag short, stepwise upper-register runs before a longer note."""
    clusters = cluster_onsets(items, tolerance=max(0.035, beat_seconds * 0.09))
    melody = []
    for cluster in clusters:
        upper = [item for item in cluster if item["note"].pitch >= split_pitch]
        candidates = upper or cluster
        melody.append(max(candidates, key=lambda item: item["note"].pitch))

    ornaments = []
    i = 0
    while i < len(melody) - 2:
        run = [melody[i]]
        j = i + 1
        while j < len(melody):
            previous = run[-1]["note"]
            current = melody[j]["note"]
            gap = current.start - previous.start
            pitch_step = abs(current.pitch - previous.pitch)
            if (gap > beat_seconds * 0.42 or previous.end - previous.start > beat_seconds * 0.42
                    or current.end - current.start > beat_seconds * 0.42 or pitch_step > 5
                    or current.start - run[0]["note"].start > beat_seconds * 1.5):
                break
            run.append(melody[j])
            j += 1
        following = melody[j] if j < len(melody) else None
        stepwise = len(run) >= 3 and sum(
            abs(run[k]["note"].pitch - run[k - 1]["note"].pitch) <= 4
            for k in range(1, len(run))
        ) >= len(run) - 2
        if (stepwise and following is not None
                and following["note"].start - run[-1]["note"].start <= beat_seconds * 0.65
                and following["note"].end - following["note"].start >= beat_seconds * 0.55):
            ornaments.append({
                "start_seconds": round(run[0]["note"].start, 4),
                "end_seconds": round(run[-1]["note"].end, 4),
                "note_ids": [item["id"] for item in run],
                "pitches_midi": [item["note"].pitch for item in run],
                "principal_note_id": following["id"],
                "confidence": "candidate",
            })
            i = j
        else:
            i += max(1, len(run) - 1)
    return ornaments


def build_score_midi(items: list[dict], output_path: Path, bpm: float, meter: tuple[int, int],
                     beat_seconds: float, phase: float, swing_ratio: float, grid_divisions: int,
                     omit_ids: set[int] | None = None) -> None:
    omit_ids = omit_ids or set()
    minimum_length = 1.0 / grid_divisions
    mapped = []
    for item in items:
        if item["id"] in omit_ids:
            continue
        note = item["note"]
        start_beat = nearest_grid_beat(note.start, beat_seconds, phase, swing_ratio, grid_divisions)
        end_beat = nearest_grid_beat(note.end, beat_seconds, phase, swing_ratio, grid_divisions)
        if end_beat <= start_beat:
            end_beat = start_beat + 1.0 / grid_divisions
        mapped.append((item, start_beat, end_beat))

    if not mapped:
        if omit_ids:
            raise RuntimeError("ornament simplification would remove every note")
        result = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
        result.time_signature_changes.append(pretty_midi.TimeSignature(meter[0], meter[1], time=0.0))
        result.instruments.append(pretty_midi.Instrument(program=0, name="Piano"))
        result.write(str(output_path))
        return

    beats_per_bar = meter[0] * 4.0 / meter[1]
    first_beat = min(start for _, start, _ in mapped)
    origin_beat = math.floor(first_beat / beats_per_bar) * beats_per_bar
    result = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    result.time_signature_changes.append(pretty_midi.TimeSignature(meter[0], meter[1], time=0.0))
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for item, start_beat, end_beat in mapped:
        note = item["note"]
        start = max(0.0, (start_beat - origin_beat) * beat_seconds)
        end = max(start + minimum_length * beat_seconds, (end_beat - origin_beat) * beat_seconds)
        piano.notes.append(pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=start,
            end=end,
        ))
    result.instruments.append(piano)
    result.write(str(output_path))


def create_score_outputs(midi_path: Path, output_dir: Path, title: str, bpm: float,
                         meter_text: str, swing_ratio: float, grid_divisions: int,
                         beat_offset: float | None, split_pitch: int,
                         key_text: str | None = None, musescore: str | None = None,
                         skip_pdf: bool = False) -> dict:
    if bpm < 40 or bpm > 300:
        raise ValueError("BPM must be between 40 and 300")
    if grid_divisions not in (2, 4, 8):
        raise ValueError("grid divisions must be 2, 4, or 8 per beat")
    meter = parse_meter(meter_text)
    if grid_divisions % 2:
        raise ValueError("grid divisions must be even to represent swing eighths")

    performance = pretty_midi.PrettyMIDI(str(midi_path))
    items = []
    for instrument_index, instrument in enumerate(performance.instruments):
        for note_index, note in enumerate(instrument.notes):
            items.append({"id": len(items), "instrument": instrument_index,
                          "source_index": note_index, "note": note})
    if not items:
        raise RuntimeError("Transkun MIDI contains no note events")

    beat_seconds = 60.0 / bpm
    onset_clusters = cluster_onsets(items, tolerance=0.045)
    cluster_times = [float(np.median([item["note"].start for item in cluster])) for cluster in onset_clusters]
    phase = beat_offset if beat_offset is not None else fit_beat_phase(
        cluster_times, beat_seconds, swing_ratio, grid_divisions
    )
    ornaments = detect_ornaments(items, beat_seconds, split_pitch)
    ornament_ids = {note_id for group in ornaments for note_id in group["note_ids"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    score_mid = output_dir / "score.mid"
    simplified_mid = output_dir / "score_simplified.mid"
    ornaments_mid = output_dir / "ornaments.mid"
    full_xml = output_dir / "score.musicxml"
    simplified_xml = output_dir / "score_simplified.musicxml"
    full_pdf = output_dir / "score.pdf"
    simplified_pdf = output_dir / "score_simplified.pdf"

    build_score_midi(items, score_mid, bpm, meter, beat_seconds, phase, swing_ratio, grid_divisions)
    build_score_midi(items, simplified_mid, bpm, meter, beat_seconds, phase, swing_ratio,
                     grid_divisions, omit_ids=ornament_ids)
    # Separate candidate ornament notes into their own playable MIDI track/file.
    ornament_items = [item for item in items if item["id"] in ornament_ids]
    build_score_midi(ornament_items, ornaments_mid, bpm, meter, beat_seconds, phase,
                     swing_ratio, grid_divisions)
    ornaments_json = output_dir / "ornaments.json"
    ornaments_json.write_text(json.dumps({
        "method": "conservative upper-register short-run heuristic; review before deleting",
        "bpm": bpm,
        "meter": meter_text,
        "swing_ratio": f"{swing_ratio}:1",
        "groups": ornaments,
    }, indent=2) + "\n", encoding="utf-8")

    swing_text = f"Swing {swing_ratio:g}:1" if swing_ratio > 1.01 else None
    if key_text is None:
        from music21 import converter
        inferred_key = converter.parse(str(score_mid)).analyze("key")
        key_text = f"{inferred_key.tonic.name} {inferred_key.mode}"

    make_musicxml(score_mid, full_xml, title, meter_text, 0, split_pitch,
                  swing_text=swing_text, tempo_bpm=bpm, key_text=key_text)
    make_musicxml(simplified_mid, simplified_xml, title + " (simplified)", meter_text, 0, split_pitch,
                  swing_text=swing_text, tempo_bpm=bpm, key_text=key_text)

    if not skip_pdf:
        if musescore is None:
            musescore = next((shutil.which(name) for name in ("musescore", "mscore", "MuseScore4")
                              if shutil.which(name)), None)
        if not musescore:
            raise RuntimeError("MuseScore not found; install it or pass --musescore")
        render_pdf(full_xml, full_pdf, musescore)
        render_pdf(simplified_xml, simplified_pdf, musescore)

    residuals = []
    for cluster_time in cluster_times:
        quantized = nearest_grid_beat(cluster_time, beat_seconds, phase, swing_ratio, grid_divisions)
        reconstructed = phase + quantized * beat_seconds
        residuals.append(abs(cluster_time - reconstructed))
    diagnostics = {
        "input_note_count": len(items),
        "ornament_candidate_notes": len(ornament_ids),
        "ornament_candidate_groups": len(ornaments),
        "bpm": bpm,
        "meter": meter_text,
        "swing_ratio": f"{swing_ratio}:1",
        "key": key_text,
        "beat_offset_seconds": round(phase, 4),
        "median_onset_adjustment_ms": round(float(np.median(residuals)) * 1000, 1),
        "p90_onset_adjustment_ms": round(float(np.quantile(residuals, 0.9)) * 1000, 1),
        "score_mid": str(score_mid),
        "simplified_mid": str(simplified_mid),
        "ornaments_json": str(ornaments_json),
        "full_musicxml": str(full_xml),
        "simplified_musicxml": str(simplified_xml),
        "full_pdf": str(full_pdf) if not skip_pdf else None,
        "simplified_pdf": str(simplified_pdf) if not skip_pdf else None,
    }
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beat-grid and swing-aware cleanup of Transkun piano MIDI")
    parser.add_argument("midi", type=Path, help="Transkun performance MIDI; it is never modified")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Piano transcription")
    parser.add_argument("--bpm", type=float, help="quarter-note BPM; omit to estimate with librosa")
    parser.add_argument("--meter", default="4/4", help="score meter (default 4/4; verify for each piece)")
    parser.add_argument("--swing-ratio", default="1:1", help="long:short eighth ratio, e.g. 2:1; use 1:1 for straight")
    parser.add_argument("--key", help="key signature override, e.g. 'C major'; default is inferred")
    parser.add_argument("--grid-divisions", type=int, default=4, help="notation grid divisions per beat: 2, 4, or 8")
    parser.add_argument("--beat-offset", type=float, help="manual beat-zero offset in seconds; default fits onset grid")
    parser.add_argument("--split-pitch", type=int, default=60)
    parser.add_argument("--musescore", help="MuseScore executable/path")
    parser.add_argument("--skip-pdf", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        bpm = args.bpm
        if bpm is None:
            wav = args.output_dir / "audio.wav"
            if not wav.is_file():
                parser.error("--bpm is required unless audio.wav exists in --output-dir for tempo estimation")
            bpm = estimate_bpm(wav)
            print(f"Estimated tempo: {bpm:.1f} BPM (verify; beat trackers can choose half/double time).", file=sys.stderr)
        ratio = parse_swing_ratio(args.swing_ratio)
        create_score_outputs(
            args.midi.expanduser().resolve(), args.output_dir.expanduser().resolve(), args.title,
            bpm, args.meter, ratio, args.grid_divisions, args.beat_offset, args.split_pitch,
            args.key, args.musescore, args.skip_pdf,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
