#!/usr/bin/env python3
"""Standalone CLI inference for Edge0/Audio8-TTS-Realistic-Air."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    return torch.device(value)


def resolve_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_model = Path(
        os.environ.get("AUDIO8_MODEL_DIR", str(base_dir / "models" / "Audio8-TTS-Realistic-Air"))
    )

    parser = argparse.ArgumentParser(description="Synthesize speech with Edge0/Audio8-TTS-Realistic-Air")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", type=Path, default=Path("output.wav"), help="Output WAV path")
    parser.add_argument("--reference-audio", type=Path, default=None, help="Optional reference WAV for voice cloning")
    parser.add_argument("--reference-text", default=None, help="Exact transcript of --reference-audio")
    parser.add_argument("--model-dir", type=Path, default=default_model, help="Local model directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (deterministic)")
    args = parser.parse_args()

    if (args.reference_audio is None) != (args.reference_text is None):
        parser.error("--reference-audio and --reference-text must be provided together")

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Run download_model.py first.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(device)

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        )
        .eval()
        .to(device)
    )

    kwargs: dict[str, object] = {
        "text": [args.text],
        "return_tensors": "pt",
    }
    if args.reference_audio is not None:
        if not args.reference_audio.is_file():
            raise FileNotFoundError(f"Reference audio not found: {args.reference_audio}")
        kwargs["reference_audio"] = [str(args.reference_audio)]
        kwargs["reference_text"] = [args.reference_text]

    inputs = processor(**kwargs)
    inputs = {name: val.to(device) for name, val in inputs.items()}

    generator = torch.Generator(device=device.type).manual_seed(seed)
    t0 = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.greedy,
        generator=generator,
        return_dict_in_generate=True,
    )
    elapsed = time.perf_counter() - t0

    code_len = int(output.code_lengths[0])
    if code_len <= 0:
        raise RuntimeError("Model produced 0 audio frames")

    codes = output.codes[0, :, :code_len]
    waveforms, waveform_lengths = model.decode_audio(codes)
    wav_len = int(waveform_lengths[0])
    waveform = waveforms[0, :wav_len].float().cpu().numpy()

    sample_rate = int(getattr(model.config.codec_config, "sample_rate", 44100))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), np.asarray(waveform, dtype=np.float32), sample_rate)

    duration = len(waveform) / sample_rate
    rtf = elapsed / duration if duration > 0 else 0.0
    finished = bool(output.finished[0])
    print(
        f"Saved {args.output} ({duration:.2f}s @ {sample_rate} Hz, "
        f"elapsed={elapsed:.2f}s, rtf={rtf:.3f}, frames={code_len}, finished={finished}, seed={seed})"
    )


if __name__ == "__main__":
    main()
