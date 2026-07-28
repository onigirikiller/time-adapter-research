from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LABEL_DIR = ROOT / "data/omni3b_delayed_backchannel_v2"
DEFAULT_OUT_DIR = ROOT / "data/omni3b_delayed_backchannel_v2_qwen3tts_clean_0p6b"
QWEN3TTS_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN3TTS_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_fragment(text: str) -> str:
    text = re.sub(r"\s*\(case\s+[^)]*\)\s*$", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def trim_audio(wav: np.ndarray, sr: int, threshold: float = 0.008, pad_s: float = 0.05) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return wav
    idx = np.flatnonzero(np.abs(wav) > threshold)
    if idx.size == 0:
        return wav
    pad = int(round(sr * pad_s))
    lo = max(0, int(idx[0]) - pad)
    hi = min(len(wav), int(idx[-1]) + pad)
    return wav[lo:hi].astype(np.float32)


def fade_and_limit(wav: np.ndarray, sr: int, max_s: float = 16.0) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    limit = int(round(sr * max_s))
    if wav.size > limit:
        wav = wav[:limit].copy()
    fade = min(int(round(sr * 0.04)), max(1, wav.size // 4))
    if wav.size > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.98:
        wav = wav / peak * 0.98
    return wav.astype(np.float32)


def load_qwen3tts(model_id: str):
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation="eager",
    )


def generate_audio(model, model_id: str, text: str, out_path: Path, speaker: str, language: str, instruct: str, force: bool) -> dict:
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and out_path.stat().st_size > 4096 and meta_path.exists() and not force:
        return read_json(meta_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavs, sr = model.generate_custom_voice(
        text=[text],
        language=[language],
        speaker=[speaker],
        instruct=[instruct],
        max_new_tokens=420,
    )
    wav = fade_and_limit(trim_audio(np.asarray(wavs[0], dtype=np.float32), int(sr)), int(sr))
    sf.write(str(out_path), wav, int(sr))
    meta = {
        "path": str(out_path),
        "model_id": model_id,
        "source": "Qwen3TTS generate_custom_voice",
        "speaker": speaker,
        "language": language,
        "instruct": instruct,
        "text": text,
        "sample_rate": int(sr),
        "duration_seconds": float(len(wav) / max(int(sr), 1)),
        "trim": "amplitude threshold 0.008 with 50ms pad",
        "waveform_silence": "No silence is appended in this clean base-audio dataset. Time enters through external timer features.",
    }
    write_json(meta_path, meta)
    return meta


def build_rows(source_rows: list[dict], context_audio: dict[str, dict], out_dir: Path) -> list[dict]:
    out = []
    for row in source_rows:
        clean = clean_fragment(row["fragment"])
        audio_meta = context_audio[row["context_id"]]
        new_row = dict(row)
        new_row["fragment_original"] = row["fragment"]
        new_row["fragment"] = clean
        new_row["audio_path_original"] = row["audio_path"]
        new_row["audio_path"] = audio_meta["path"]
        new_row["sample_rate"] = audio_meta["sample_rate"]
        new_row["speech_duration_seconds"] = audio_meta["duration_seconds"]
        new_row["total_duration_seconds"] = audio_meta["duration_seconds"]
        new_row["clean_qwen3tts_audio"] = True
        new_row["qwen3tts_model_id"] = audio_meta["model_id"]
        new_row["audio_timing_semantics"] = "base speech only; row silence_seconds is external timer value, not appended waveform silence"
        out.append(new_row)
    return out


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    source_manifest = read_json(SOURCE_LABEL_DIR / "manifest.json")
    splits = {split: read_jsonl(SOURCE_LABEL_DIR / f"{split}.jsonl") for split in ["train", "validation", "test"]}
    contexts = {}
    for split, rows in splits.items():
        for row in rows:
            if row["context_id"] not in contexts:
                contexts[row["context_id"]] = {
                    "context_id": row["context_id"],
                    "split": split,
                    "profile": row["profile"],
                    "fragment": clean_fragment(row["fragment"]),
                }
    selected_contexts = list(contexts.values())
    if args.max_contexts:
        selected_contexts = selected_contexts[: args.max_contexts]
    selected_ids = {ctx["context_id"] for ctx in selected_contexts}
    model_id = QWEN3TTS_17B if args.model_size == "1.7b" else QWEN3TTS_06B
    model = load_qwen3tts(model_id)
    context_audio = {}
    started = time.time()
    for i, ctx in enumerate(selected_contexts):
        out_path = audio_dir / f"{ctx['context_id']}_base_qwen3tts.wav"
        meta = generate_audio(
            model,
            model_id,
            ctx["fragment"],
            out_path,
            args.speaker,
            args.language,
            args.instruct,
            args.force,
        )
        context_audio[ctx["context_id"]] = meta
        if (i + 1) % args.log_every == 0 or i == 0 or i + 1 == len(selected_contexts):
            elapsed = time.time() - started
            print(f"generated/reused {i + 1}/{len(selected_contexts)} contexts elapsed={elapsed:.1f}s", flush=True)
    if args.max_contexts:
        splits = {
            split: [row for row in rows if row["context_id"] in selected_ids]
            for split, rows in splits.items()
        }
    clean_splits = {split: build_rows(rows, context_audio, out_dir) for split, rows in splits.items()}
    for split, rows in clean_splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_label_dataset": str(SOURCE_LABEL_DIR),
        "source_label_manifest": source_manifest,
        "qwen3tts_model_id": model_id,
        "speaker": args.speaker,
        "language": args.language,
        "instruct": args.instruct,
        "context_count": len(selected_contexts),
        "split_counts": {split: len(rows) for split, rows in clean_splits.items()},
        "label_counts": {split: dict(Counter(row["label"] for row in rows)) for split, rows in clean_splits.items()},
        "profile_counts": {split: dict(Counter(row["profile"] for row in rows)) for split, rows in clean_splits.items()},
        "fragment_cleaning": "Removed trailing '(case ...)' synthetic management markers.",
        "audio_policy": "One clean Qwen3TTS base speech file per context. No waveform silence is appended; silence_seconds remains an external timer feature.",
        "audio_dir": str(audio_dir),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote {out_dir / 'manifest.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-size", choices=["0.6b", "1.7b"], default="0.6b")
    parser.add_argument("--speaker", default="serena")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default="Speak naturally and conversationally, with clear audio and no extra commentary.")
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
