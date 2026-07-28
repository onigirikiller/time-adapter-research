from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/drive_test_audio_2026-06-28"
OUT_DIR = ROOT / "artifacts/omni3b_single_token_lora_drive_audio_vad_demo_v1"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = import_module(ROOT / "scripts/run_single_token_lora_vad_realtime_demo_v2.py", "single_token_vad_realtime_v2")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def real_audio_spec(path: Path) -> dict:
    stem = path.stem
    profile = "asked_wait" if "WAIT" in stem.upper() else "vulnerable"
    return {
        "title": f"Real user audio from Drive: {path.name}",
        "profile": profile,
        "chunks": [],
        "backchannel_instruction": (
            "Give only a very short acknowledgement in the same language as the user's audio if clear. "
            "Do not take the turn."
        ),
        "support_instruction": (
            "Give one brief supportive response in the same language as the user's audio if clear. "
            "Do not mention labels, timing, or system details."
        ),
    }


def speech_intervals_from_vad(frames: list[dict], duration_s: float, frame_s: float, hop_s: float) -> list[dict]:
    intervals: list[dict] = []
    cur = None
    for f in frames:
        if not f.get("vad_speaking"):
            continue
        end = min(duration_s, float(f["time_s"]))
        start = max(0.0, end - frame_s)
        if cur is not None and start <= cur["speech_end_s"] + hop_s * 1.5:
            cur["speech_end_s"] = max(cur["speech_end_s"], end)
        else:
            cur = {
                "chunk_index": len(intervals),
                "text": "VAD speech region",
                "speech_start_s": start,
                "speech_end_s": end,
                "silence_after_s": 0.0,
                "silence_start_s": end,
                "silence_end_s": end,
            }
            intervals.append(cur)
    return intervals


def prepare_audio(path: Path, out_dir: Path) -> tuple[np.ndarray, int, Path, dict]:
    wav, sr = v2.read_audio(path)
    wav, sr = v2.resample_to_24k(wav, sr)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.98:
        wav = wav / peak * 0.98
    out_path = out_dir / "input_drive_audio_24k.wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav.astype(np.float32), sr)
    meta = {
        "source_path": str(path),
        "path": str(out_path),
        "sample_rate": int(sr),
        "duration_s": float(len(wav) / sr),
        "method": "User-provided Drive WAV, converted to mono 24 kHz only. No synthetic speech or extra silence added.",
    }
    return wav.astype(np.float32), sr, out_path, meta


def build_talker_prompt(action: str, spec: dict) -> str:
    instruction = spec["backchannel_instruction"] if action == "BACKCHANNEL" else spec["support_instruction"]
    return (
        "You are the listener in a spoken dialogue.\n"
        "The user's audio so far is attached. No transcript is provided.\n"
        f"The timing controller selected {action}.\n"
        f"{instruction}\n"
        "Return only the words to speak aloud. "
        "Do not mention WAIT, BACKCHANNEL, SUPPORT, labels, timers, versions, or system details. "
        "Do not use parentheses, brackets, markdown, stage directions, or quoted metadata."
    )


def generate_talker_response(model, processor, spec: dict, action: str, out_path: Path, speaker: str, context_audio_path: Path, force: bool = False) -> dict:
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and out_path.stat().st_size > 4096 and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    prompt = build_talker_prompt(action, spec)
    conv = [
        {"role": "system", "content": [{"type": "text", "text": v2.v4.DEFAULT_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(context_audio_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = v2.v4.move_inputs(inputs, model.device, model.dtype)
    max_text = 20 if action == "BACKCHANNEL" else 48
    max_audio_s = 2.0 if action == "BACKCHANNEL" else 5.0
    attempts = []
    selected = None
    for attempt in range(3):
        torch.manual_seed(20260628 + attempt)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260628 + attempt)
        with torch.inference_mode():
            result = model.generate(
                **inputs,
                return_audio=True,
                speaker=speaker,
                thinker_max_new_tokens=max_text,
                talker_max_new_tokens=768,
                talker_do_sample=True,
                talker_top_k=40,
                talker_top_p=0.8,
                talker_temperature=0.9,
                talker_repetition_penalty=1.05,
                use_audio_in_video=False,
            )
        text_ids, audio = result
        decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        assistant_text = decoded.split("assistant\n")[-1].strip()
        raw_wav = audio.detach().float().cpu().numpy().reshape(-1)
        trimmed = v2.v4.trim_audio(raw_wav, 24000, threshold=0.006)
        limited = len(trimmed) > int(max_audio_s * 24000)
        wav = v2.v4.fade_and_limit(trimmed, 24000, max_s=max_audio_s)
        metrics = v2.v4.audio_metrics(wav, 24000)
        rejected = v2.v4.invalid_spoken_text(assistant_text) or v2.v4.suspicious_talker_audio(metrics, action, limited)
        attempts.append({"attempt": attempt, "assistant_text": assistant_text, "limited": limited, "metrics": metrics, "rejected": rejected})
        selected = (assistant_text, wav, metrics)
        if not rejected:
            break
    assistant_text, wav, metrics = selected
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, 24000)
    meta = {
        "action": action,
        "speaker": speaker,
        "prompt": prompt,
        "context_audio_path": str(context_audio_path),
        "assistant_text": assistant_text,
        "sample_rate": 24000,
        "duration_s": len(wav) / 24000,
        "metrics": metrics,
        "attempts": attempts,
        "path": str(out_path),
        "source": "Qwen2.5-Omni-3B Talker",
    }
    write_json(meta_path, meta)
    return meta


def generate_response_events(model, processor, spec: dict, speech_decisions: list[dict], out_dir: Path, speaker: str, force: bool) -> list[dict]:
    events = []
    for i, decision in enumerate(speech_decisions):
        action = decision["raw_label"]
        out_path = out_dir / "responses" / f"{i:02d}_{int(round(decision['clock_s'] * 1000)):06d}ms_{action.lower()}_{speaker.lower()}.wav"
        meta = generate_talker_response(model, processor, spec, action, out_path, speaker, Path(decision["segment_audio_path"]), force=force)
        events.append(
            {
                "event_index": i,
                "action": action,
                "clock_s": float(decision["clock_s"]),
                "silence_elapsed": float(decision["features"]["silence_elapsed"]),
                "audio_path": meta["path"],
                "assistant_text": meta["assistant_text"],
                "speaker": speaker,
                "duration_s": meta["metrics"]["duration_s"],
                "source_segment": decision["segment_audio_path"],
            }
        )
    return events


def mark_future_overlap(events: list[dict], speech_intervals: list[dict]):
    for event in events:
        start = float(event["clock_s"])
        end = float(event.get("actual_audio_end_s", start))
        event["overlaps_future_user_speech"] = any(
            float(item["speech_start_s"]) < end and float(item["speech_end_s"]) > start + 0.05
            for item in speech_intervals
        )


def process_audio_files(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(INPUT_DIR.glob("*.wav"))
    if args.file:
        wavs = [INPUT_DIR / args.file]
    prepared = {}
    for path in wavs:
        if not path.exists():
            raise FileNotFoundError(path)
        key = path.stem
        out_dir = OUT_DIR / key
        spec = real_audio_spec(path)
        stream, sr, input_path, audio_meta = prepare_audio(path, out_dir)
        rows, vad_meta = v2.build_vad_rows(stream, sr, spec, args.vad_threshold, args.tick)
        rows = v2.write_segments(stream, sr, rows, out_dir / "segments")
        speech_intervals = speech_intervals_from_vad(
            vad_meta["frames"],
            duration_s=float(len(stream) / sr),
            frame_s=float(vad_meta["frame_s"]),
            hop_s=float(vad_meta["hop_s"]),
        )
        stream_meta = {
            **audio_meta,
            "demo_title": "DirectLM Real-User-Audio VAD Demo",
            "input_description": "Drive user WAV. Fixed RMS-VAD triggers DirectLM ticks; Omni receives audio prefix only.",
            "viewer_text": f"Viewer note: real user audio file {path.name}; transcript is not provided to the classifier.",
            "timeline": speech_intervals,
            "timeline_legend": "blue=VAD speech regions, gaps=below-threshold audio, vertical lines=VAD-triggered DirectLM ticks",
            "drive_source": {
                "folder": "test audio",
                "file_name": path.name,
                "note": "Input audio was user-provided and already noise-reduced.",
            },
        }
        write_json(out_dir / "vad_rows.json", {"rows": rows, "vad": {k: v for k, v in vad_meta.items() if k != "frames"}})
        prepared[key] = {
            "path": path,
            "spec": spec,
            "out_dir": out_dir,
            "input_path": input_path,
            "stream": stream,
            "sr": sr,
            "stream_meta": stream_meta,
            "rows": rows,
            "vad_meta": vad_meta,
            "speech_intervals": speech_intervals,
        }

    processor, classifier, dtype, token_ids, adapter = v2.load_classifier()
    for key, item in prepared.items():
        decisions = v2.classify_rows(processor, classifier, dtype, token_ids, adapter, item["rows"])
        item["decisions"] = decisions
        write_csv(item["out_dir"] / "per_tick_decisions.csv", decisions)
    del classifier, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    talker_model, talker_processor = v2.load_talker()
    summaries = {}
    for key, item in prepared.items():
        speech_decisions = v2.select_speech_events(item["decisions"])
        if args.max_events > 0:
            speech_decisions = speech_decisions[: args.max_events]
        events = generate_response_events(talker_model, talker_processor, item["spec"], speech_decisions, item["out_dir"], args.speaker, args.force_talker)
        audio_meta = v2.mix_audio(item["stream"], item["sr"], events, item["out_dir"] / "demo_mix.wav")
        mark_future_overlap(events, item["speech_intervals"])
        render_meta = {**item["stream_meta"], "duration_s": max(float(item["stream_meta"]["duration_s"]), float(audio_meta["duration_s"]))}
        video = v2.render_video(
            item["spec"],
            render_meta,
            item["decisions"],
            item["vad_meta"],
            events,
            Path(audio_meta["path"]),
            item["out_dir"] / "demo_real_user_audio_vad_directlm.mp4",
        )
        summary = {
            "spec": item["spec"],
            "input_audio": str(item["input_path"]),
            "stream_meta": item["stream_meta"],
            "vad": {k: v for k, v in item["vad_meta"].items() if k != "frames"},
            "decisions": item["decisions"],
            "events": events,
            "mixed_audio": audio_meta,
            "video": str(video),
            "model": {
                "base": v2.MODEL_ID,
                "lora": str(v2.BEST_LORA),
                "codebook": v2.CODEBOOK,
                "layer": v2.LAYER,
                "alpha": v2.ALPHA,
                "prompt_mode": "audio_only",
                "classifier_input": "audio segment from stream start to current VAD tick only",
                "talker": "Qwen2.5-Omni-3B Talker",
            },
        }
        write_json(item["out_dir"] / "summary.json", summary)
        summaries[key] = {"video": str(video), "events": events, "num_ticks": len(item["decisions"])}
        print(f"Wrote {video}", flush=True)

    write_json(OUT_DIR / "summary.json", summaries)
    zip_path = OUT_DIR / "drive_test_audio_real_user_vad_demo_v1_2026-06-28.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in OUT_DIR.rglob("*"):
            if path.is_file() and path != zip_path:
                z.write(path, path.relative_to(OUT_DIR))
    print(f"Wrote {zip_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="")
    parser.add_argument("--speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--vad-threshold", type=float, default=0.01)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--force-talker", action="store_true")
    args = parser.parse_args()
    process_audio_files(args)


if __name__ == "__main__":
    main()
