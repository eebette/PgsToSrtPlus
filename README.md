# PgsToSrtPlus

Extract PGS (Blu-ray) subtitles from MKV files and convert them to SRT using OCR.

## How It Works

PgsToSrtPlus decodes PGS subtitle bitmaps, preprocesses and splits them into individual text lines, then runs a two-stage OCR pipeline:

1. **PaddleOCR** performs a fast first-pass recognition on every line. Lines above a confidence threshold are accepted as-is.
2. Lines below the threshold fall back to a **Vision Language Model** (via Ollama) for a more accurate read.
3. A second VLM pass performs **italic detection** by comparing the original subtitle bitmap against a synthetically rendered upright reference image, classifying each token as italic or roman.

The result is an SRT file with accurate text and `<i>` markup.

## Supported Languages

- **English** (`en`) — default
- **Japanese** (`ja`)

Other languages may work but will use a generic fallback prompt. Language-specific OCR prompts, fonts, and post-processing steps are configurable per language.

## Requirements

- **Docker**
- **Ollama** running separately and accessible from the Docker container (used for low-confidence OCR fallback and italic detection). Default model: `qwen3-vl:32b-instruct`

## Quick Start

Pull the image:

```bash
# CPU
docker pull ebette1/pgs-to-srt-plus:latest

# GPU (NVIDIA)
docker pull ebette1/pgs-to-srt-plus-gpu:latest
```

Run:

```bash
docker run --rm --add-host=host.docker.internal:host-gateway \
  -v /path/to/media:/media \
  ebette1/pgs-to-srt-plus:latest \
  "/media/movie.mkv" \
  --ollama http://host.docker.internal:11434
```

For GPU acceleration (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker run --rm --gpus all --add-host=host.docker.internal:host-gateway \
  -v /path/to/media:/media \
  ebette1/pgs-to-srt-plus-gpu:latest \
  "/media/movie.mkv" \
  --ollama http://host.docker.internal:11434
```

The SRT file is written next to the input file. Use `-o /path` with a bind mount to write elsewhere.

## Options

| Option | Default | Description |
|---|---|---|
| `--ollama` | `http://127.0.0.1:11434` | Ollama endpoint URL |
| `--language`, `-l` | `en` | Subtitle language (`en`, `ja`) |
| `--track` | auto-detect | PGS track index |
| `-o`, `--output` | same as input | Output directory |
| `--model` | `qwen3-vl:32b-instruct` | Ollama VLM model |
| `--device` | `cpu` | PaddleOCR device (`cpu`, `gpu`) |
| `--verify-threshold` | `0.97` | PaddleOCR confidence below which to fall back to VLM |
| `--paddle-model` | `PP-OCRv5_server_rec` | PaddleOCR recognition model |

## Acknowledgments

- [Tentacule/PgsToSrt](https://github.com/Tentacule/PgsToSrt) — the original inspiration for this project
- [SubtitleEdit / libse](https://github.com/SubtitleEdit/subtitleedit) — PGS parsing and Matroska container support

## Docker Images

- [ebette1/pgs-to-srt-plus](https://hub.docker.com/r/ebette1/pgs-to-srt-plus) (CPU)
- [ebette1/pgs-to-srt-plus-gpu](https://hub.docker.com/r/ebette1/pgs-to-srt-plus-gpu) (NVIDIA GPU)
