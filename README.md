# Audio8-TTS Realistic-Air Web UI

One-click **Web UI**, **REST API**, **standalone CLI**, and **NixOS systemd module** for [**Edge0/Audio8-TTS-Realistic-Air**](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) (0.6B DualAR PyTorch speech synthesis & zero-shot voice cloning, 23 languages, 44.1 kHz audio).

- **Zero-dependency single-page Web UI** (`index.html`) served by **FastAPI** (`server.py`)
- **One-click setup & launch** on **Linux/macOS** (`./run.sh`), **Windows** (`run.bat`), and **remote GPU servers over SSH** (`./remote.sh`)
- **Zero-shot voice cloning** from uploaded audio (`WAV`, `FLAC`, `OGG`, `MP3`, `M4A`, `WebM`) or live browser microphone recording
- **Saved voice library** + **recent generations history** (auto-pruned to 50 outputs)
- **Live Showcase Samples** included in [`samples/`](./samples/README.md)

---

## Showcase Audio

| Sample | Mode | Duration | Generation Time / RTF (RTX 3060 12GB) | Files |
| :--- | :--- | :--- | :--- | :--- |
| **01 — Default Voice** | Zero-shot (no reference) | `15.19s` | `11.77s` (`RTF 0.775`) | [`01-zero-shot.wav`](./samples/01-zero-shot.wav) · [`01-zero-shot.json`](./samples/01-zero-shot.json) |
| **02 — Voice Clone** | Zero-shot voice clone | `14.91s` | `12.17s` (`RTF 0.817`) | [`02-voice-clone.wav`](./samples/02-voice-clone.wav) · [`02-voice-clone.json`](./samples/02-voice-clone.json) |
| **03 — Ye Grand ISP (IVR)** | 4-chunk voice-consistent (676 chars) | `50.93s` | `39.64s` (`RTF 0.778`) | [`03-ye-grand-isp.wav`](./samples/03-ye-grand-isp.wav) · [`03-ye-grand-isp.json`](./samples/03-ye-grand-isp.json) |
| **Reference Clip** | 5.48s source voice | `5.48s` | — | [`reference-chicanery.wav`](./samples/reference-chicanery.wav) |

Remote cold-start + SSH tunnel startup took `3.64s`. See [`samples/README.md`](./samples/README.md) for exact prompts, transcripts, seeds, per-chunk breakdown, and CLI reproduction commands.

---

## One-Click Quick Start

### Linux / macOS

```bash
git clone https://github.com/IAFahim/audio8-tts-webui.git
cd audio8-tts-webui
./run.sh
```

`./run.sh` automatically:
1. Creates a Python 3.11 `.venv` (prefers `uv`, falls back to `python3 -m venv`).
2. Detects NVIDIA GPU via `nvidia-smi` and installs PyTorch `2.5.0+cu124` (or CPU PyTorch if no GPU is present) + `requirements.txt`.
3. Injects `.venv`'s `nvidia/*/lib` (`libnvrtc.so.12`, `libcudnn`, etc.) into `LD_LIBRARY_PATH` on Linux so CUDA kernels work out of the box even on NixOS or minimal distros.
4. Downloads `Edge0/Audio8-TTS-Realistic-Air` (~2.6 GB, skipping the unused 3.76 GB GGUF directory) into `./models/Audio8-TTS-Realistic-Air`.
5. Starts `uvicorn` on `http://127.0.0.1:8999` and opens your browser automatically.

Useful flags:
- `./run.sh --cpu` — Force CPU mode (`float32`)
- `./run.sh --port 9000 --no-browser` — Custom port without opening a browser
- `./run.sh --download-only` — Fetch weights and exit

### Windows

Double-click **`run.bat`** in Explorer, or run from Command Prompt:

```bat
git clone https://github.com/IAFahim/audio8-tts-webui.git
cd audio8-tts-webui
run.bat
```

### Remote GPU Server Over SSH (One-Click Tunnel)

If your GPU lives on another Linux machine running `audio8-tts.service` (see [`nixos/audio8-tts-module.nix`](./nixos/audio8-tts-module.nix)):

```bash
cp .env.example .env
# Edit .env with AUDIO8_REMOTE_HOST and AUDIO8_REMOTE_CONTROL
./remote.sh
```

- Starts the remote systemd service, opens `ssh -L 127.0.0.1:8999:127.0.0.1:8999`, waits for `/api/health`, and opens `http://127.0.0.1:8999` (secure origin so browser microphone recording works).
- Press `Ctrl+C` (or run `./remote.sh stop`) to close the tunnel and stop the remote service so GPU VRAM is freed automatically.
- Use `./remote.sh --detach` to return your terminal immediately and `./remote.sh status` to inspect tunnel/service state.

---

## Standalone CLI (`infer.py`)

```bash
# Default voice
.venv/bin/python infer.py \
  --text "Hello! This is Audio8-TTS Realistic-Air running locally." \
  --output hello.wav

# Zero-shot voice cloning
.venv/bin/python infer.py \
  --reference-audio samples/reference-chicanery.wav \
  --reference-text "And I saved him. And I shouldn't have. I took him into my own firm." \
  --text "Voice cloning with an exact reference transcript." \
  --output cloned.wav
```

---

## HTTP API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Single-page Web UI |
| `GET` | `/api/health` | Model readiness, GPU name, VRAM stats, active queue status |
| `GET` | `/api/voices` | List saved voice profiles |
| `POST` | `/api/voices` | Upload/record & normalize (44.1 kHz mono WAV) a new voice profile |
| `GET` | `/api/voices/{id}/audio` | Stream saved voice WAV |
| `DELETE` | `/api/voices/{id}` | Delete a saved voice profile |
| `POST` | `/api/synthesize` | Synthesize speech (`text`, optional `voice_id` or `reference_audio` + `reference_text`, sampling params) |
| `GET` | `/api/outputs` | List recent generations (up to 50) |
| `GET` | `/api/outputs/{id}.wav` | Stream or download (`?download=1`) generated WAV |
| `DELETE` | `/api/outputs/{id}` | Delete a generated clip |

---

## NixOS Module

A sanitized NixOS module is included at [`nixos/audio8-tts-module.nix`](./nixos/audio8-tts-module.nix) configuring an on-demand `audio8-tts.service` (`wantedBy = [ ]`), `audio8` system user, `ffmpeg-headless`, and automatic `nvidia/*/lib` `LD_LIBRARY_PATH` wrapping.

---

## Notes & Requirements

- **Python 3.10–3.12** (3.11 recommended).
- **`ffmpeg`** is optional but recommended if you want to upload or record compressed formats (`MP3`, `M4A`, `WebM`). `WAV`, `FLAC`, and `OGG` work without `ffmpeg` via `soundfile` + `torchaudio`.
- **`transformers>=4.57.0,<5`** is pinned in `requirements.txt` for `ArkttsModel` compatibility.

## License & Attribution

- Repository code and Web UI are licensed under the **Apache License 2.0** ([`LICENSE`](./LICENSE), [`NOTICE`](./NOTICE)).
- Upstream model & architecture: [**Edge0/Audio8-TTS**](https://github.com/Edge0-AI/Audio8_TTS) / [**Edge0/Audio8-TTS-Realistic-Air**](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air), inspired by the DualAR architecture in [**Fish Audio S2 Pro**](https://github.com/fishaudio/fish-speech).
