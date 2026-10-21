#!/usr/bin/env python3
"""Solo-piano audio/video -> MIDI -> MusicXML -> PDF pipeline."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


SAMPLE_RATE = 44_100


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def run(command: list[str]) -> None:
    print("+", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


def download_audio(url: str, work_dir: Path) -> Path:
    template = work_dir / "source.%(ext)s"
    run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--no-progress",
            "--format",
            "bestaudio/best",
            "--output",
            str(template),
            url,
        ]
    )
    downloads = [p for p in work_dir.glob("source.*") if p.is_file() and not p.name.endswith(".part")]
    if not downloads:
        raise RuntimeError("yt-dlp completed but no downloaded audio file was found")
    return downloads[0]


def extract_wav(source: Path, wav_path: Path, ffmpeg: str) -> None:
    if source.resolve() == wav_path.resolve():
        raise RuntimeError("input and normalized audio output resolve to the same file; choose a different --output-dir")
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a usable WAV: {wav_path}")


def transcribe(wav_path: Path, midi_path: Path, device: str) -> None:
    run(
        [
            sys.executable,
            "-m",
            "transkun.transcribe",
            str(wav_path),
            str(midi_path),
            "--device",
            device,
        ]
    )
    if not midi_path.is_file() or midi_path.stat().st_size == 0:
        raise RuntimeError(f"Transkun did not create a usable MIDI file: {midi_path}")


def make_musicxml(
    midi_path: Path,
    musicxml_path: Path,
    title: str,
    meter_text: str,
    quantize_subdivisions: int,
    split_pitch: int,
    swing_text: str | None = None,
    tempo_bpm: float | None = None,
    key_text: str | None = None,
) -> None:
    try:
        from music21 import chord, clef, converter, expressions, instrument, key, layout, meter, metadata, note, pitch, stream, tempo
    except ImportError as exc:
        raise RuntimeError("music21 is not installed; run `uv sync` in the project directory") from exc

    parsed = converter.parse(str(midi_path))
    # Bucket simultaneous notes of equal duration into chords for readable notation.
    buckets: dict[tuple[str, float, float], list[object]] = defaultdict(list)
    grid = 1.0 / quantize_subdivisions if quantize_subdivisions else None

    def snap(value: float) -> float:
        return round(value / grid) * grid if grid else value

    for event in parsed.recurse().notes:
        if isinstance(event, chord.Chord):
            pitches = list(event.pitches)
        elif isinstance(event, note.Note):
            pitches = [event.pitch]
        else:
            continue

        start = float(event.getOffsetInHierarchy(parsed))
        end = start + float(event.duration.quarterLength)
        start = snap(start)
        end = snap(end)
        if end <= start:
            end = start + (grid or 1.0 / 16.0)

        for event_pitch in pitches:
            staff = "bass" if event_pitch.midi < split_pitch else "treble"
            buckets[(staff, round(start, 7), round(end, 7))].append(event_pitch)

    if not buckets:
        raise RuntimeError("Transkun produced a MIDI file with no notes")

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    if key_text:
        match = re.fullmatch(r"\s*([A-Ga-g](?:#|b)?)\s*(major|minor)?\s*", key_text, flags=re.IGNORECASE)
        if not match:
            raise RuntimeError("key must be a tonic with optional mode, e.g. 'C major' or 'Bb major'")
        score_key = key.Key(match.group(1), (match.group(2) or "major").lower())
    else:
        try:
            score_key = parsed.analyze("key")
        except Exception:
            score_key = None

    staff_parts = {}
    staff_voices = {}
    for staff_name, staff_clef in (("treble", clef.TrebleClef()), ("bass", clef.BassClef())):
        part = stream.Part()
        part.partName = "Piano"
        part.insert(0, instrument.Piano())
        part.insert(0, staff_clef)
        part.insert(0, meter.TimeSignature(meter_text))
        if score_key is not None:
            part.insert(0, key.KeySignature(score_key.sharps))
        staff_parts[staff_name] = part
        staff_voices[staff_name] = stream.Voice(id="1")

    for (staff_name, start, end), pitches in buckets.items():
        unique_pitches = sorted({p.nameWithOctave for p in pitches}, key=lambda name: pitch.Pitch(name).midi)
        if len(unique_pitches) == 1:
            event = note.Note(unique_pitches[0])
        else:
            event = chord.Chord(unique_pitches)
        event.duration.quarterLength = end - start
        staff_voices[staff_name].insert(start, event)

    if swing_text:
        staff_parts["treble"].insert(0, expressions.TextExpression(swing_text))
    if tempo_bpm is not None:
        staff_parts["treble"].insert(0, tempo.MetronomeMark(number=tempo_bpm))

    for staff_name, part in staff_parts.items():
        part.insert(0, staff_voices[staff_name])
        part.makeNotation(inPlace=True)
        score.insert(0, part)

    # Preserve source tempo marks only if the score stage did not set a fixed tempo.
    if tempo_bpm is None:
        for mark in parsed.recurse().getElementsByClass(tempo.MetronomeMark):
            score.insert(float(mark.getOffsetInHierarchy(parsed)), mark)

    score.insert(
        0,
        layout.StaffGroup(
            [staff_parts["treble"], staff_parts["bass"]],
            name="Piano",
            symbol="brace",
            barTogether=True,
        ),
    )
    score.write("musicxml", fp=str(musicxml_path))


def find_musescore(explicit: str | None) -> str | None:
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        if Path(explicit).is_file():
            return explicit
        return None
    for name in ("musescore", "mscore", "MuseScore4"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None


def render_pdf(musicxml_path: Path, pdf_path: Path, executable: str) -> None:
    command = [executable, "-o", str(pdf_path), str(musicxml_path)]
    try:
        run(command)
    except subprocess.CalledProcessError as exc:
        # MuseScore 4 may refuse music21 MusicXML with exit code 40 unless forced.
        if exc.returncode != 40:
            raise
        print("MuseScore rejected the MusicXML import; retrying with -f.", file=sys.stderr)
        run([executable, "-f", "-o", str(pdf_path), str(musicxml_path)])
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"MuseScore did not create a usable PDF: {pdf_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe authorized solo-piano audio/video to MIDI, MusicXML, and PDF."
    )
    parser.add_argument("input", help="local audio/video path or URL to a video you are authorized to process")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for all pipeline outputs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Transkun inference device")
    parser.add_argument("--meter", default="4/4", help="score time signature; verify/override for each piece (default: 4/4)")
    parser.add_argument("--score-bpm", type=float, help="score tempo; omit to estimate from audio (verify half/double-time errors)")
    parser.add_argument("--swing-ratio", default="1:1", help="long:short eighth ratio, e.g. 2:1; 1:1 is straight")
    parser.add_argument("--key", help="key signature override, e.g. 'C major'; default is inferred from MIDI")
    parser.add_argument("--grid-divisions", type=int, default=4, help="score grid divisions per beat: 2, 4, or 8")
    parser.add_argument("--beat-offset", type=float, help="manual beat-zero offset in seconds; default fits the note onsets")
    parser.add_argument(
        "--split-pitch",
        type=int,
        default=60,
        metavar="MIDI_NOTE",
        help="notes below this MIDI pitch go on the bass staff; default 60 (middle C)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg on PATH)")
    parser.add_argument("--musescore", help="MuseScore executable/path; otherwise search PATH")
    parser.add_argument("--skip-pdf", action="store_true", help="stop after writing MusicXML")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.grid_divisions not in (2, 4, 8):
        parser.error("--grid-divisions must be 2, 4, or 8")
    if args.split_pitch < 0 or args.split_pitch > 127:
        parser.error("--split-pitch must be a MIDI note number from 0 to 127")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio.wav"
    midi_path = output_dir / "transcription.mid"
    scoreify_path = Path(__file__).with_name("scoreify.py")
    input_path = Path(args.input).expanduser()
    title = input_path.stem if not is_url(args.input) else "Piano transcription"

    try:
        if is_url(args.input):
            with tempfile.TemporaryDirectory(prefix="piano-source-") as temporary_dir:
                downloaded = download_audio(args.input, Path(temporary_dir))
                extract_wav(downloaded, wav_path, args.ffmpeg)
        else:
            if not input_path.is_file():
                parser.error(f"input file does not exist: {input_path}")
            extract_wav(input_path.resolve(), wav_path, args.ffmpeg)

        transcribe(wav_path, midi_path, args.device)
        score_command = [
            sys.executable,
            str(scoreify_path),
            str(midi_path),
            "--output-dir",
            str(output_dir),
            "--title",
            title,
            "--meter",
            args.meter,
            "--swing-ratio",
            args.swing_ratio,
            "--grid-divisions",
            str(args.grid_divisions),
            "--split-pitch",
            str(args.split_pitch),
        ]
        if args.score_bpm is not None:
            score_command.extend(["--bpm", str(args.score_bpm)])
        if args.key:
            score_command.extend(["--key", args.key])
        if args.beat_offset is not None:
            score_command.extend(["--beat-offset", str(args.beat_offset)])
        if args.musescore:
            score_command.extend(["--musescore", args.musescore])
        if args.skip_pdf:
            score_command.append("--skip-pdf")
        run(score_command)

        print(f"WAV:               {wav_path}")
        print(f"Performance MIDI:  {midi_path}")
        print(f"Score outputs:     {output_dir}")
    except subprocess.CalledProcessError as exc:
        print(f"Pipeline command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
