# Solo piano transcription pipeline

A local pipeline for videos/audio you own or are authorized to process:

```text
URL or local audio/video
  -> yt-dlp (URL only)
  -> ffmpeg mono WAV at 44.1 kHz
  -> Transkun performance MIDI (kept unchanged)
  -> beat-grid/swing score pass
  -> full and ornament-simplified MIDI, MusicXML, and optional MuseScore PDFs
```

For solo piano this deliberately does **not** denoise or run source separation. Those steps can remove note attacks/harmonics or add separation artifacts.

## Requirements

- Python 3.10 or 3.11 (Transkun/PyTorch compatibility; this project excludes Python 3.12+)
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH`
- MuseScore 4 on `PATH` for PDF output (or provide its executable with `--musescore`). On some Linux installs the executable is named `mscore` (for example `/usr/bin/mscore`), not `musescore`.

Install dependencies:

```bash
uv sync --python 3.11
```

The default inference device is CPU. For CUDA, install a CUDA-enabled PyTorch build compatible with your GPU, verify `torch.cuda.is_available()`, and pass `--device cuda`.

## Run

For a piece with a steady tempo, grid mode is an optional alternative. The example uses a 2:1 swing ratio as a starting point; adjust it for the performance.

```bash
uv run python pipeline.py 'VIDEO_URL' \
  --output-dir output/piano-recording \
  --score-bpm 120 --meter 4/4 --swing-ratio 2:1
```

For other pieces, omit `--score-bpm` to estimate it from the audio, but check for half/double-time errors. Tempo estimates can be half or double time, so verify them. Meter is not automatically inferred; the default is 4/4. Use `--swing-ratio 1:1` for straight timing. `--key` overrides the automatically inferred key signature.

The score pass can also be run again **without downloading or transcribing** from an existing Transkun MIDI:

```bash
uv run python scoreify.py output/piano-recording/transcription.mid \
  --output-dir output/piano-recording \
  --bpm 120 --meter 4/4 --swing-ratio 2:1
```

## Outputs

- `audio.wav` — normalized source audio supplied to Transkun
- `transcription.mid` — original Transkun performance MIDI; never modified by score cleanup
- `score.mid` — beat-grid/swing-normalized MIDI, all notes retained
- `score_simplified.mid` — same score with conservative ornament candidates removed
- `ornaments.mid` / `ornaments.json` — candidate ornament notes in an isolated MIDI plus a reviewable list; heuristic candidates can be false positives
- `score.musicxml` / `score.pdf` — full score
- `score_simplified.musicxml` / `score_simplified.pdf` — simplified alternative

`--skip-pdf` omits both PDF renders. Use `--help` on either script for all controls.

## Score cleanup controls and limitations

- `--score-bpm` sets the quarter-note tempo. The audio beat tracker can choose half or double time; manual overrides are recommended for known pieces.
- `--meter 3/4` sets the time signature; default `4/4` is only a fallback, not meter inference.
- `--swing-ratio 2:1` maps swung eighth offbeats to conventional even eighth-note notation and adds a swing marking. `1:1` keeps a straight grid. This is a constant-tempo/constant-swing first pass; it does not model rubato or changing swing feel bar by bar.
- The key signature is inferred from the transcription unless overridden with `--key 'C major'`; key analysis can be wrong on modal or highly chromatic music.
- `--grid-divisions 4` quantizes to sixteenth-note score positions per beat; choose 2 for eighths or 8 for 32nds. The beat phase is fitted from MIDI note onsets unless `--beat-offset` is supplied.
- Ornament detection is deliberately conservative and heuristic. It identifies short upper-line runs before longer notes, not definitive grace notes/trills. Review `ornaments.json`; compare full and simplified PDFs before using the simplified score.
- Grand-staff assignment remains a pitch heuristic: notes below `--split-pitch 60` go to bass staff; notes at/above it go to treble. It does not infer hands or handle hand-crossing intelligently. Review the output in MuseScore.

The score pass is an editable prototype, not an automatic professional engraving system. Preserve the performance MIDI and review meter, beat phase, quantization, ornaments, and staff assignment against the recording.
