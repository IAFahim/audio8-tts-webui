#!/usr/bin/env python3
"""Download Edge0/Audio8-TTS-Realistic-Air weights without unused GGUF files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "Edge0/Audio8-TTS-Realistic-Air"
REQUIRED_FILES = (
    "model.safetensors",
    "codec.pth",
    "tokenizer.json",
    "modeling_arktts.py",
)


def is_model_complete(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in REQUIRED_FILES)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_dir = Path(
        os.environ.get("AUDIO8_MODEL_DIR", str(base_dir / "models" / "Audio8-TTS-Realistic-Air"))
    )

    parser = argparse.ArgumentParser(description="Download Audio8-TTS Realistic-Air weights")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face model repo ID")
    parser.add_argument("--dir", type=Path, default=default_dir, help="Target local directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args()

    model_dir = args.dir.resolve()
    if not args.force and is_model_complete(model_dir):
        print(f"[download_model] Model already present at {model_dir}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_model] Downloading {args.repo} -> {model_dir} (skipping llama-cpp/** GGUFs)...")
    snapshot_download(
        args.repo,
        local_dir=str(model_dir),
        ignore_patterns=["llama-cpp/**"],
    )

    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Download completed, but required files are missing: {', '.join(missing)}")

    print(f"[download_model] Verified model files in {model_dir}")


if __name__ == "__main__":
    main()
