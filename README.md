# Solo piano transcription pipeline

Transcribe authorized solo-piano recordings into performance MIDI, then generate full and simplified MusicXML/PDF scores. Timing is preserved by default; an experimental grid-based pass is optional.

The pipeline accepts video/audio you own or are authorized to process:

```text
URL or local audio/video
  -> yt-dlp (URL only)
  -> ffmpeg mono WAV at 44.1 kHz
  -> Transkun performance MIDI (kept unchanged)
  -> timing-preserving full/simplified scores (default)
  -> optional explicit grid-quantized alternative
```

For solo piano this deliberately does **not** denoise or run source separation. Source separation was evaluated on test material and removed real notes, so it is not part of the supported pipeline.

## Requirements

- Python 3.10 or 3.11 (Transkun/PyTorch compatibility; this project excludes Python 3.12+)
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH`
- MuseScore 4 on `PATH` for PDF output (or provide its executable with `--musescore`). On some Linux installs the executable is named `mscore` (for example `/usr/bin/mscore`), not `musescore`.

For MIDI-to-score conversion only, install the base dependencies:

```bash
uv sync --python 3.11
```

For Transkun audio transcription, add the `transcription` extra. Add `youtube` when the input is a URL rather than a local file:

```bash
uv sync --python 3.11 --extra transcription --extra youtube
```

Automatic tempo estimation in grid mode also requires the `tempo-estimation` extra. It is not needed when you provide `--bpm`:

```bash
uv sync --python 3.11 --extra transcription --extra youtube --extra tempo-estimation
```

The default inference device is CPU. For CUDA, install a CUDA-enabled PyTorch build compatible with your GPU, verify `torch.cuda.is_available()`, and pass `--device cuda`.

## Run

The default score pass preserves Transkun's performed note onsets and lengths; it does not impose a fixed tempo or swing ratio:

For a local recording:

```bash
uv run --no-sync python pipeline.py /path/to/piano-recording.wav \
  --output-dir output/piano-recording --meter 4/4
```

For a video URL, install the `youtube` extra and replace `VIDEO_URL` with a URL you are authorized to process:

```bash
uv run --no-sync python pipeline.py 'VIDEO_URL' \
  --output-dir output/piano-recording --meter 4/4
```

The score pass can be rerun from an existing Transkun MIDI without downloading or transcribing again:

```bash
uv run --no-sync python scoreify.py output/piano-recording/transcription.mid \
  --output-dir output/piano-recording --timing-mode preserve --meter 4/4
```

An experimental fixed-grid alternative is opt-in. It uses one global tempo and may shift note events; only use it when that is desired:

```bash
uv run --no-sync python scoreify.py output/piano-recording/transcription.mid \
  --output-dir output/grid-test --timing-mode grid \
  --bpm 120 --meter 4/4 --swing-ratio 2:1
```

## Outputs

- `audio.wav` — normalized source audio supplied to Transkun
- `transcription.mid` — original Transkun performance MIDI; never modified
- `score.mid` — in preserve mode, byte-for-byte copy of the performance MIDI
- `score_simplified.mid` — ornament-candidate notes removed, with all remaining note times preserved
- `ornaments.mid` / `ornaments.json` — candidate ornament notes isolated at their original times, plus a reviewable list; heuristic candidates can be false positives
- `score.musicxml` / `score.pdf` — full score
- `score_simplified.musicxml` / `score_simplified.pdf` — simplified alternative

`--skip-pdf` omits both PDF renders. Use `--help` on either script for all controls.

## Timing and notation controls

- `--timing-mode preserve` is the default. It preserves performance MIDI timing and does not emit a claimed BPM or swing marking. The resulting score may still have intricate notation because expressive timing is retained.
- `--timing-mode grid` is experimental. It requires `--bpm` or the `tempo-estimation` extra to estimate tempo from `audio.wav`, then snaps note events to a global grid. Beat trackers can select half/double time, so verify estimates rather than treating them as ground truth. `--swing-ratio` only affects this opt-in mode.
- `--meter 4/4` sets the score meter; the default is 4/4, but meter is not automatically inferred.
- The key signature is inferred from the transcription unless overridden with `--key 'C major'`; key analysis can be wrong on modal or highly chromatic music.
- Ornament detection is deliberately conservative and heuristic. It identifies short upper-line runs before longer notes, not definitive grace notes/trills. Review `ornaments.json`; compare full and simplified scores before using the simplified one.
- Grand-staff assignment remains a pitch heuristic: notes below `--split-pitch 60` go to bass staff; notes at/above it go to treble. It does not infer hands or handle hand-crossing intelligently.

The score pass is an editable prototype, not an automatic professional engraving system. Preserve the performance MIDI and review the meter, notation, ornaments, and staff assignment against the recording.

## License

No license has been selected for this repository yet. Until one is added, assume the code is not licensed for reuse or redistribution.
