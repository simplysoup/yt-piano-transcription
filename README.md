# Solo piano transcription pipeline

A local command-line pipeline for videos/audio you own or are authorized to process:

```text
URL or local audio/video
  -> yt-dlp (URL only)
  -> ffmpeg mono WAV at 44.1 kHz
  -> Transkun piano transcription
  -> MIDI
  -> music21 two-staff MusicXML
  -> MuseScore PDF (optional)
```

For solo piano this deliberately does **not** denoise or run source separation. Those steps can remove note attacks/harmonics or add separation artifacts. The normalized audio, MIDI, and MusicXML are kept in the output directory.

## Requirements

- Python 3.10 or 3.11 (Transkun/PyTorch compatibility; this project currently excludes Python 3.12+)
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH`
- MuseScore 4 on `PATH` for PDF output (or provide its executable with `--musescore`). On some Linux installs the executable is named `mscore` (for example `/usr/bin/mscore`), not `musescore`.

Install the Python dependencies:

```bash
uv sync --python 3.11
```

The default inference device is CPU. To use CUDA, install a CUDA-enabled PyTorch build compatible with your GPU in the project environment, verify `torch.cuda.is_available()`, then pass `--device cuda`.

## Run

YouTube URL (only download/process videos you own or are authorized to use):

```bash
uv run python pipeline.py 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --output-dir output/my-piece
```

Local audio or video:

```bash
uv run python pipeline.py ./my-piece.mp4 --output-dir output/my-piece
```

Run Transkun on a CUDA-enabled PyTorch installation:

```bash
uv run python pipeline.py ./my-piece.wav --output-dir output/my-piece --device cuda
```

If MuseScore is unavailable, create WAV, MIDI, and MusicXML only:

```bash
uv run python pipeline.py ./my-piece.wav --output-dir output/my-piece --skip-pdf
```

Use `uv run python pipeline.py --help` to see all options. The output files are:

- `audio.wav` — mono, 44.1 kHz PCM audio supplied to Transkun
- `transcription.mid` — Transkun's note transcription
- `score.musicxml` — notation generated from the MIDI
- `score.pdf` — MuseScore rendering, unless `--skip-pdf` is used

## Notation controls and limitations

- `--meter 3/4` sets the time signature used for engraving. Default: `4/4`.
- `--quantize 4` snaps note starts and ends to a sixteenth-note grid (1/4 quarter note). Use `--quantize 8` for a finer grid or `--quantize 0` to disable snapping. Quantization can simplify expressive timing, but may also move notes to incorrect rhythmic positions.
- The initial grand-staff split is a pitch heuristic: MIDI notes below `--split-pitch 60` go to bass staff; notes at/above it go to treble. It does not infer hands, handle hand-crossing intelligently, or reconstruct polished rhythms, key signatures, beams, or pedal markings. Review the MusicXML in MuseScore and adjust as needed.
- Automatic audio-to-score transcription is not fully reliable. Preserve the MIDI and WAV as intermediate results and review the output against the recording.
