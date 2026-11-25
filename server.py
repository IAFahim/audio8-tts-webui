#!/usr/bin/env python3
"""Standalone FastAPI backend and Web UI server for Edge0/Audio8-TTS-Realistic-Air."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from transformers import AutoModel, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("AUDIO8_MODEL_DIR", str(BASE_DIR / "models" / "Audio8-TTS-Realistic-Air"))
).resolve()
DATA_DIR = Path(os.environ.get("AUDIO8_DATA_DIR", str(BASE_DIR / "data"))).resolve()
WEBUI_DIR = Path(os.environ.get("AUDIO8_WEBUI_DIR", str(BASE_DIR))).resolve()
DEVICE_OVERRIDE = os.environ.get("AUDIO8_DEVICE", "auto").strip().lower() or "auto"

VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"
INDEX_HTML = WEBUI_DIR / "index.html"

SAMPLE_RATE_HZ = 44100
MIN_REF_SECONDS = 0.5
MAX_REF_SECONDS = 60.0
MAX_SAVED_OUTPUTS = 50
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class AppState:
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.dtype_name: str = "float32"
        self.gpu_name: Optional[str] = None
        self.sample_rate: int = SAMPLE_RATE_HZ
        self.ready: bool = False
        self.load_error: Optional[str] = None
        self.lock: asyncio.Lock = asyncio.Lock()


state = AppState()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-_")
    return (cleaned[:40] if cleaned else "voice").lower()


def _validate_safe_id(item_id: str, label: str = "id") -> str:
    if not item_id or not SAFE_ID_RE.match(item_id):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return item_id


def _resolve_under(base_dir: Path, filename: str) -> Path:
    base_resolved = base_dir.resolve()
    target = (base_resolved / filename).resolve()
    if target.parent != base_resolved:
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


def _normalize_audio_to_wav(src_path: Path, dst_wav_path: Path) -> float:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(src_path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE_HZ),
            "-c:a",
            "pcm_s16le",
            str(dst_wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "ffmpeg failed").strip().splitlines()[-1]
            raise HTTPException(status_code=400, detail=f"Audio decode failed: {tail}")
        try:
            audio, sr = sf.read(str(dst_wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read normalized WAV: {exc}") from exc
        if sr != SAMPLE_RATE_HZ:
            raise HTTPException(status_code=400, detail=f"Unexpected sample rate after normalize: {sr}")
    else:
        try:
            audio_np, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio decode failed without ffmpeg ({exc}). "
                    "Install ffmpeg to decode MP3/M4A/WebM, or upload a WAV/FLAC/OGG file."
                ),
            ) from exc
        tensor = torch.from_numpy(np.asarray(audio_np, dtype=np.float32))
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=-1)
        if sr != SAMPLE_RATE_HZ:
            tensor = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE_HZ)
        audio = tensor.contiguous().numpy()
        sf.write(str(dst_wav_path), audio, SAMPLE_RATE_HZ, subtype="PCM_16")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    duration_sec = float(len(audio) / SAMPLE_RATE_HZ) if SAMPLE_RATE_HZ > 0 else 0.0
    if duration_sec < MIN_REF_SECONDS or duration_sec > MAX_REF_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Reference audio duration ({duration_sec:.2f}s) must be between {MIN_REF_SECONDS} and {MAX_REF_SECONDS:.0f}s",
        )
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0
    if peak < 1e-4 or rms < 1e-5:
        raise HTTPException(status_code=400, detail="Reference audio appears silent")
    return duration_sec


def _list_json_items(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    items: list[dict[str, Any]] = []
    for meta_path in directory.glob("*.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            item_id = data.get("id")
            if not item_id:
                continue
            wav_path = _resolve_under(directory, f"{item_id}.wav")
            if not wav_path.exists():
                continue
            items.append(data)
        except Exception:
            continue
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return items


def _prune_old_outputs(max_keep: int = MAX_SAVED_OUTPUTS) -> None:
    items = _list_json_items(OUTPUTS_DIR)
    for stale in items[max_keep:]:
        item_id = stale.get("id")
        if not item_id or not SAFE_ID_RE.match(str(item_id)):
            continue
        for ext in (".wav", ".json"):
            p = OUTPUTS_DIR / f"{item_id}{ext}"
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


def _resolve_runtime_device() -> tuple[str, torch.dtype, str, Optional[str]]:
    if DEVICE_OVERRIDE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("AUDIO8_DEVICE=cuda requested, but torch.cuda.is_available() is False")
        use_cuda = True
    elif DEVICE_OVERRIDE == "cpu":
        use_cuda = False
    elif DEVICE_OVERRIDE == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        raise RuntimeError(f"Unsupported AUDIO8_DEVICE value: {DEVICE_OVERRIDE!r} (expected auto, cuda, cpu)")

    if use_cuda:
        return "cuda", torch.bfloat16, "bfloat16", torch.cuda.get_device_name(0)
    return "cpu", torch.float32, "float32", None


def _load_model_sync() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. Run `python download_model.py` first."
        )
    device, dtype, dtype_name, gpu_name = _resolve_runtime_device()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        )
        .eval()
        .to(device)
    )
    state.processor = processor
    state.model = model
    state.device = device
    state.dtype = dtype
    state.dtype_name = dtype_name
    state.gpu_name = gpu_name
    state.sample_rate = int(getattr(model.config.codec_config, "sample_rate", SAMPLE_RATE_HZ))
    state.ready = True
    state.load_error = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(_load_model_sync)
    except Exception as exc:
        state.ready = False
        state.load_error = str(exc)
        raise
    yield
    if state.model is not None:
        state.model = None
        state.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="Audio8-TTS Realistic-Air WebUI", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    vram_alloc_mb = 0.0
    vram_res_mb = 0.0
    vram_total_mb = 0.0
    if state.device == "cuda" and torch.cuda.is_available():
        vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        vram_res_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
        vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
    return {
        "ready": state.ready,
        "load_error": state.load_error,
        "model_id": "Edge0/Audio8-TTS-Realistic-Air",
        "model_dir": str(MODEL_DIR),
        "device": state.device,
        "gpu_name": state.gpu_name,
        "dtype": state.dtype_name,
        "sample_rate": state.sample_rate,
        "vram_allocated_mb": vram_alloc_mb,
        "vram_reserved_mb": vram_res_mb,
        "vram_total_mb": vram_total_mb,
        "busy": state.lock.locked(),
        "voices_count": len(_list_json_items(VOICES_DIR)),
        "outputs_count": len(_list_json_items(OUTPUTS_DIR)),
    }


@app.get("/api/voices")
def list_voices() -> list[dict[str, Any]]:
    return _list_json_items(VOICES_DIR)


@app.post("/api/voices")
async def create_voice(
    name: str = Form(...),
    reference_text: str = Form(...),
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    clean_name = name.strip()
    clean_ref_text = reference_text.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Voice name is required")
    if not clean_ref_text:
        raise HTTPException(status_code=400, detail="Exact reference transcript is required")

    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    voice_id = f"{_slugify(clean_name)}-{uuid.uuid4().hex[:6]}"
    dest_wav = _resolve_under(VOICES_DIR, f"{voice_id}.wav")
    dest_meta = _resolve_under(VOICES_DIR, f"{voice_id}.json")

    suffix = Path(audio.filename or "audio.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        duration_sec = await asyncio.to_thread(_normalize_audio_to_wav, tmp_path, dest_wav)
    except Exception:
        if dest_wav.exists():
            dest_wav.unlink()
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    meta = {
        "id": voice_id,
        "name": clean_name,
        "reference_text": clean_ref_text,
        "duration_sec": round(duration_sec, 2),
        "created_at": _utc_iso(),
        "audio_url": f"/api/voices/{voice_id}/audio",
    }
    dest_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


@app.get("/api/voices/{voice_id}/audio")
def get_voice_audio(voice_id: str) -> FileResponse:
    safe_id = _validate_safe_id(voice_id, "voice_id")
    wav_path = _resolve_under(VOICES_DIR, f"{safe_id}.wav")
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    return FileResponse(path=wav_path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict[str, str]:
    safe_id = _validate_safe_id(voice_id, "voice_id")
    wav_path = _resolve_under(VOICES_DIR, f"{safe_id}.wav")
    meta_path = _resolve_under(VOICES_DIR, f"{safe_id}.json")
    if not wav_path.exists() and not meta_path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    if wav_path.exists():
        wav_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    return {"status": "deleted", "id": safe_id}


def _synthesize_sync(
    *,
    text: str,
    ref_wav_path: Optional[str],
    ref_text: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    greedy: bool,
    auto_retry: bool,
) -> dict[str, Any]:
    if not state.ready or state.model is None or state.processor is None:
        raise RuntimeError("Model is not loaded")

    processor_kwargs: dict[str, Any] = {
        "text": [text],
        "return_tensors": "pt",
    }
    if ref_wav_path is not None and ref_text is not None:
        processor_kwargs["reference_audio"] = [ref_wav_path]
        processor_kwargs["reference_text"] = [ref_text]

    inputs = state.processor(**processor_kwargs)
    inputs = {name: val.to(state.device) for name, val in inputs.items()}

    def run_once(token_budget: int) -> tuple[Any, np.ndarray, int, bool]:
        generator = torch.Generator(device=state.device).manual_seed(seed)
        output = state.model.generate(
            **inputs,
            max_new_tokens=token_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=not greedy,
            generator=generator,
            return_dict_in_generate=True,
        )
        code_len = int(output.code_lengths[0])
        if code_len <= 0:
            raise RuntimeError("Model produced 0 audio frames")
        codes = output.codes[0, :, :code_len]
        waveforms, waveform_lengths = state.model.decode_audio(codes)
        wav_len = int(waveform_lengths[0])
        waveform_np = waveforms[0, :wav_len].float().cpu().numpy()
        finished = bool(output.finished[0])
        return output, waveform_np, code_len, finished

    t0 = time.perf_counter()
    try:
        _, waveform_np, code_len, finished = run_once(max_new_tokens)
        if not finished and auto_retry:
            retry_budget = max(max_new_tokens, 2000)
            if retry_budget > max_new_tokens:
                _, waveform_np, code_len, finished = run_once(retry_budget)
    finally:
        if state.device == "cuda":
            torch.cuda.empty_cache()

    elapsed_sec = time.perf_counter() - t0
    return {
        "waveform": waveform_np,
        "code_frames": code_len,
        "finished": finished,
        "elapsed_sec": elapsed_sec,
    }


@app.post("/api/synthesize")
async def synthesize(
    text: str = Form(...),
    voice_id: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(50),
    max_new_tokens: int = Form(1024),
    seed: Optional[str] = Form(None),
    greedy: bool = Form(False),
    auto_retry: bool = Form(True),
) -> dict[str, Any]:
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Input text must not be empty")
    if not (0.05 <= temperature <= 1.5):
        raise HTTPException(status_code=400, detail="temperature must be between 0.05 and 1.5")
    if not (0.05 <= top_p <= 1.0):
        raise HTTPException(status_code=400, detail="top_p must be between 0.05 and 1.0")
    if not (0 <= top_k <= 200):
        raise HTTPException(status_code=400, detail="top_k must be between 0 and 200")
    if not (64 <= max_new_tokens <= 2000):
        raise HTTPException(status_code=400, detail="max_new_tokens must be between 64 and 2000")

    if seed is not None and seed.strip() != "":
        try:
            used_seed = int(seed.strip())
            if used_seed < 0:
                raise ValueError("negative seed")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="seed must be a non-negative integer") from exc
    else:
        used_seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

    ref_wav_path: Optional[str] = None
    ref_text_val: Optional[str] = None
    voice_name = "No Reference (Default)"
    temp_files: list[Path] = []

    try:
        if reference_audio is not None and reference_audio.filename:
            raw_bytes = await reference_audio.read()
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="Uploaded quick-clone audio is empty")
            clean_ref_text = (reference_text or "").strip()
            if not clean_ref_text:
                raise HTTPException(status_code=400, detail="Reference transcript is required for quick clone")

            suffix = Path(reference_audio.filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as raw_tmp:
                raw_tmp.write(raw_bytes)
                raw_tmp_path = Path(raw_tmp.name)
            temp_files.append(raw_tmp_path)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_tmp_path = Path(wav_tmp.name)
            temp_files.append(wav_tmp_path)

            await asyncio.to_thread(_normalize_audio_to_wav, raw_tmp_path, wav_tmp_path)
            ref_wav_path = str(wav_tmp_path)
            ref_text_val = clean_ref_text
            voice_name = f"Quick Clone ({reference_audio.filename})"
        elif voice_id and voice_id.strip():
            safe_voice_id = _validate_safe_id(voice_id.strip(), "voice_id")
            wav_path = _resolve_under(VOICES_DIR, f"{safe_voice_id}.wav")
            meta_path = _resolve_under(VOICES_DIR, f"{safe_voice_id}.json")
            if not wav_path.exists() or not meta_path.exists():
                raise HTTPException(status_code=404, detail="Selected voice not found")
            voice_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ref_wav_path = str(wav_path)
            ref_text_val = (reference_text or "").strip() or str(voice_meta["reference_text"])
            voice_name = str(voice_meta.get("name", safe_voice_id))

        async with state.lock:
            try:
                result = await asyncio.to_thread(
                    _synthesize_sync,
                    text=clean_text,
                    ref_wav_path=ref_wav_path,
                    ref_text=ref_text_val,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    seed=used_seed,
                    greedy=greedy,
                    auto_retry=auto_retry,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        waveform_np: np.ndarray = result["waveform"]
        duration_sec = float(len(waveform_np) / state.sample_rate) if state.sample_rate > 0 else 0.0
        elapsed_sec = float(result["elapsed_sec"])
        rtf = round(elapsed_sec / duration_sec, 3) if duration_sec > 0 else 0.0

        output_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        out_wav = _resolve_under(OUTPUTS_DIR, f"{output_id}.wav")
        out_meta = _resolve_under(OUTPUTS_DIR, f"{output_id}.json")

        sf.write(str(out_wav), waveform_np, state.sample_rate)

        payload = {
            "id": output_id,
            "audio_url": f"/api/outputs/{output_id}.wav",
            "duration_sec": round(duration_sec, 2),
            "elapsed_sec": round(elapsed_sec, 2),
            "rtf": rtf,
            "code_frames": int(result["code_frames"]),
            "finished": bool(result["finished"]),
            "seed": used_seed,
            "sample_rate": state.sample_rate,
            "voice_id": voice_id.strip() if voice_id and voice_id.strip() else None,
            "voice_name": voice_name,
            "text": clean_text,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "greedy": greedy,
            "auto_retry": auto_retry,
            "created_at": _utc_iso(),
        }
        out_meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _prune_old_outputs(MAX_SAVED_OUTPUTS)
        return payload
    finally:
        for tmp_p in temp_files:
            if tmp_p.exists():
                try:
                    tmp_p.unlink()
                except OSError:
                    pass


@app.get("/api/outputs")
def list_outputs() -> list[dict[str, Any]]:
    return _list_json_items(OUTPUTS_DIR)[:MAX_SAVED_OUTPUTS]


@app.get("/api/outputs/{output_id}.wav")
def get_output_wav(output_id: str, download: int = Query(0)) -> FileResponse:
    safe_id = _validate_safe_id(output_id, "output_id")
    wav_path = _resolve_under(OUTPUTS_DIR, f"{safe_id}.wav")
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{safe_id}.wav"'
    return FileResponse(path=wav_path, media_type="audio/wav", headers=headers)


@app.delete("/api/outputs/{output_id}")
def delete_output(output_id: str) -> dict[str, str]:
    safe_id = _validate_safe_id(output_id, "output_id")
    wav_path = _resolve_under(OUTPUTS_DIR, f"{safe_id}.wav")
    meta_path = _resolve_under(OUTPUTS_DIR, f"{safe_id}.json")
    if not wav_path.exists() and not meta_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    if wav_path.exists():
        wav_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    return {"status": "deleted", "id": safe_id}
