from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEMO = ROOT / "artifacts/omni3b_control_response_lora_demo_v1"
OUT_DIR = ROOT / "artifacts/omni3b_control_response_lora_pseudorealtime_v1"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
CODEBOOK = {"WAIT": "/W", "BACKCHANNEL": "/B", "SUPPORT": "/S"}
LAYER = 3
ALPHA = 4.0
POSITION = "all_tokens"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


demo = import_module(ROOT / "scripts/run_control_response_lora_demo_v1.py", "control_response_demo_for_pseudort")
single = demo.single
v2 = demo.v2


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


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def summarize_latencies(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean_ms": statistics.mean(vals),
        "p50_ms": percentile(vals, 0.50),
        "p90_ms": percentile(vals, 0.90),
        "p95_ms": percentile(vals, 0.95),
        "p99_ms": percentile(vals, 0.99),
        "max_ms": max(vals),
        "under_250ms_rate": sum(v <= 250 for v in vals) / len(vals),
        "under_500ms_rate": sum(v <= 500 for v in vals) / len(vals),
    }


def build_cases() -> list[dict]:
    return [
        {
            "key": "real_WAITGOAL",
            "source_type": "real",
            "profile": "asked_wait",
            "audio": SOURCE_DEMO / "real_WAITGOAL/input_audio_24k.wav",
        },
        {
            "key": "real_SUPPORTGOAL_plus_2s",
            "source_type": "real",
            "profile": "vulnerable",
            "audio": SOURCE_DEMO / "real_SUPPORTGOAL_plus_2s/input_audio_24k.wav",
        },
        {
            "key": "tts_wait_text_medium",
            "source_type": "qwen3tts",
            "profile": "asked_wait",
            "audio": SOURCE_DEMO / "tts_wait_text_medium/input_audio_24k.wav",
        },
        {
            "key": "tts_support_text_medium",
            "source_type": "qwen3tts",
            "profile": "vulnerable",
            "audio": SOURCE_DEMO / "tts_support_text_medium/input_audio_24k.wav",
        },
    ]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


def prepare_case(case: dict, args) -> dict:
    stream, sr = load_audio(case["audio"])
    out_dir = OUT_DIR / case["key"]
    spec = {"title": case["key"], "profile": case["profile"], "chunks": [], "backchannel_instruction": "", "support_instruction": ""}
    rows, vad_meta = v2.build_vad_rows(stream, sr, spec, args.vad_threshold, args.tick)
    rows = v2.write_segments(stream, sr, rows, out_dir / "segments")
    built_rows = []
    for row in rows:
        built_rows.append(
            {
                **row,
                "profile": case["profile"],
                "fragment": "[audio-only; transcript withheld from Omni]",
                "label": "",
            }
        )
    return {
        **case,
        "out_dir": out_dir,
        "stream": stream,
        "duration_s": float(len(stream) / sr),
        "sr": sr,
        "rows": built_rows,
        "vad_meta": vad_meta,
    }


def classify_tick(processor, model, dtype, token_ids, adapter, row: dict, generate_event_text: bool):
    vector = single.adapter_predict(adapter, [row])[0]

    prep_start = time.perf_counter()
    prompt_inputs, prompt_len, _ = single.prepare_inputs(
        processor,
        row,
        "audio_only",
        CODEBOOK,
        label=None,
        audio_timing_mode="row_audio",
    )
    moved = single.move_inputs(prompt_inputs, model.device, dtype)
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
    handle = single.hook_module(model, LAYER).register_forward_hook(hook)
    try:
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.thinker(**moved, use_audio_in_video=False)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
    finally:
        handle.remove()

    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    label_logits = np.array([logits[token_ids[label]] for label in LABELS], dtype=np.float64)
    probs = single.softmax3(label_logits)
    raw_label = LABELS[int(np.argmax(probs))]

    gen_ms = 0.0
    generated_raw = ""
    generated_label = raw_label
    generated_response = ""
    if generate_event_text:
        gen_hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
        gen_handle = single.hook_module(model, LAYER).register_forward_hook(gen_hook)
        try:
            gen_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.thinker.generate(
                    **moved,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=False,
                    use_audio_in_video=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            gen_ms = (time.perf_counter() - gen_start) * 1000.0
        finally:
            gen_handle.remove()
        generated_raw = processor.tokenizer.decode(
            generated[0, prompt_len:].detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_label, generated_response = demo.parse_generated(generated_raw)

    stat = hook.stats[0] if hook.stats else {}
    return {
        "raw_label": raw_label,
        "generated_label": generated_label,
        "generated_response": generated_response,
        "generated_raw": generated_raw,
        "p_WAIT": float(probs[0]),
        "p_BACKCHANNEL": float(probs[1]),
        "p_SUPPORT": float(probs[2]),
        "prep_ms": prep_ms,
        "forward_ms": forward_ms,
        "label_total_ms": prep_ms + forward_ms,
        "event_generate_ms": gen_ms,
        "total_with_event_generate_ms": prep_ms + forward_ms + gen_ms,
        "time_vector_norm": stat.get("injected_norm", 0.0),
        "context_norm": stat.get("hidden_norm", 0.0),
        "time_context_cosine": stat.get("hidden_injected_cosine", 0.0),
        "hook_calls": hook.calls,
    }


def add_realtime_simulation(rows: list[dict], tick_s: float, processing_key: str):
    serial_busy = 0.0
    latest_busy = 0.0
    latest_skips = 0
    for row in rows:
        clock = float(row["clock_s"])
        proc_s = float(row[processing_key]) / 1000.0
        start = max(clock, serial_busy)
        end = start + proc_s
        row[f"serial_start_{processing_key}_s"] = start
        row[f"serial_end_{processing_key}_s"] = end
        row[f"serial_drift_{processing_key}_ms"] = max(0.0, start - clock) * 1000.0
        row[f"serial_finish_within_500ms_{processing_key}"] = int(proc_s <= tick_s)
        serial_busy = end

        if clock < latest_busy:
            row[f"latest_only_skipped_{processing_key}"] = 1
            latest_skips += 1
        else:
            row[f"latest_only_skipped_{processing_key}"] = 0
            latest_busy = clock + proc_s
    return latest_skips


def summarize_case(case: dict, rows: list[dict], tick_s: float) -> dict:
    label_skips = add_realtime_simulation(rows, tick_s, "label_total_ms")
    full_skips = add_realtime_simulation(rows, tick_s, "total_with_event_generate_ms")
    events = [r for r in rows if r.get("event_emitted")]
    return {
        "case": case["key"],
        "source_type": case["source_type"],
        "duration_s": case["duration_s"],
        "num_ticks": len(rows),
        "tick_s": tick_s,
        "pred_counts": {label: sum(1 for r in rows if r["raw_label"] == label) for label in LABELS},
        "events": [
            {
                "clock_s": r["clock_s"],
                "silence_elapsed": r["silence_seconds"],
                "raw_label": r["raw_label"],
                "generated_label": r["generated_label"],
                "generated_response": r["generated_response"],
                "label_total_ms": r["label_total_ms"],
                "event_generate_ms": r["event_generate_ms"],
                "total_with_event_generate_ms": r["total_with_event_generate_ms"],
            }
            for r in events
        ],
        "label_latency": summarize_latencies(rows, "label_total_ms"),
        "forward_latency": summarize_latencies(rows, "forward_ms"),
        "prep_latency": summarize_latencies(rows, "prep_ms"),
        "event_generation_latency": summarize_latencies(events, "event_generate_ms"),
        "single_thread_label_latest_only_skips": label_skips,
        "single_thread_full_latest_only_skips": full_skips,
        "label_serial_max_drift_ms": max((float(r["serial_drift_label_total_ms_ms"]) for r in rows), default=0.0),
        "full_serial_max_drift_ms": max((float(r["serial_drift_total_with_event_generate_ms_ms"]) for r in rows), default=0.0),
        "label_under_500ms_rate": sum(float(r["label_total_ms"]) <= tick_s * 1000.0 for r in rows) / max(1, len(rows)),
        "full_under_500ms_rate": sum(float(r["total_with_event_generate_ms"]) <= tick_s * 1000.0 for r in rows) / max(1, len(rows)),
    }


def make_audio_only_output(case: dict, rows: list[dict], talker_model, talker_processor, speaker: str, force: bool):
    events = []
    response_dir = case["out_dir"] / "responses"
    for row in rows:
        if not row.get("event_emitted"):
            continue
        response_text = str(row.get("generated_response") or "").strip()
        if not response_text:
            continue
        event_index = len(events)
        out_path = response_dir / f"{event_index:02d}_{int(round(float(row['clock_s']) * 1000)):06d}ms_{row['raw_label'].lower()}_{speaker.lower()}.wav"
        meta = demo.generate_exact_talker_audio(talker_model, talker_processor, response_text, out_path, speaker, force=force)
        events.append(
            {
                "event_index": event_index,
                "action": row["raw_label"],
                "clock_s": float(row["clock_s"]),
                "silence_elapsed": float(row["silence_seconds"]),
                "audio_path": meta["path"],
                "assistant_text": response_text,
                "talker_spoken_text": meta.get("spoken_text", response_text),
                "speaker": speaker,
                "duration_s": meta["metrics"]["duration_s"],
            }
        )
    mixed_wav = case["out_dir"] / "pseudorealtime_audio_only_mix.wav"
    audio_meta = v2.mix_audio(case["stream"], int(case["sr"]), events, mixed_wav)
    write_json(
        case["out_dir"] / "audio_only_output.json",
        {
            "input_audio": str(case["audio"]),
            "mixed_audio": audio_meta,
            "events": events,
            "note": "Audio-only pseudo-realtime output: original input plus Omni Talker responses at measured decision clocks.",
        },
    )
    return {"mixed_audio": audio_meta, "events": events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-threshold", type=float, default=0.01)
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--limit-ticks", type=int, default=0)
    parser.add_argument("--no-event-generate", action="store_true")
    parser.add_argument("--audio-only-output", action="store_true", default=True)
    parser.add_argument("--speaker", default="Chelsie")
    parser.add_argument("--force-talker", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    if args.case:
        keep = set(args.case)
        cases = [c for c in cases if c["key"] in keep]

    prepared = [prepare_case(case, args) for case in cases]
    processor, model, dtype, token_ids, adapter = demo.load_response_thinker()

    all_rows = []
    summaries = []
    decisions_by_case = {}
    for case in prepared:
        rows = case["rows"][: args.limit_ticks or None]
        decisions = []
        last = "WAIT"
        for i, row in enumerate(rows):
            should_generate = False
            result = classify_tick(processor, model, dtype, token_ids, adapter, row, generate_event_text=False)
            label_for_event = result["raw_label"]
            if not args.no_event_generate and label_for_event != "WAIT" and label_for_event != last:
                should_generate = True
                event_result = classify_tick(processor, model, dtype, token_ids, adapter, row, generate_event_text=True)
                result.update(
                    {
                        "generated_label": event_result["generated_label"],
                        "generated_response": event_result["generated_response"],
                        "generated_raw": event_result["generated_raw"],
                        "event_generate_ms": event_result["event_generate_ms"],
                        "total_with_event_generate_ms": result["label_total_ms"] + event_result["event_generate_ms"],
                    }
                )
            event_emitted = label_for_event != "WAIT" and label_for_event != last
            if label_for_event == "WAIT":
                last = "WAIT"
            else:
                last = label_for_event
            out_row = {
                "case": case["key"],
                "source_type": case["source_type"],
                "tick_index": i,
                "clock_s": row["clock_s"],
                "silence_seconds": row["silence_seconds"],
                "delta_t": row["features"]["delta_t"],
                "utterance_elapsed": row["features"]["utterance_elapsed"],
                "asr_changed": row["features"]["asr_changed"],
                "vad_rms": row.get("vad_rms"),
                "vad_threshold": row.get("vad_threshold"),
                **result,
                "event_emitted": int(event_emitted),
                "event_text_generated": int(should_generate),
            }
            decisions.append(out_row)
            all_rows.append(out_row)
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"{case['key']} {i + 1}/{len(rows)}", flush=True)
        write_csv(case["out_dir"] / "pseudorealtime_ticks.csv", decisions)
        decisions_by_case[case["key"]] = decisions
        summaries.append(summarize_case(case, decisions, args.tick))
        write_json(case["out_dir"] / "pseudorealtime_summary.json", summaries[-1])

    audio_outputs = {}
    if args.audio_only_output:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        talker_model, talker_processor = demo.load_talker()
        for case in prepared:
            audio_outputs[case["key"]] = make_audio_only_output(
                case,
                decisions_by_case[case["key"]],
                talker_model,
                talker_processor,
                args.speaker,
                args.force_talker,
            )

    overall = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vad_threshold": args.vad_threshold,
        "tick_s": args.tick,
        "cases": summaries,
        "overall_label_latency": summarize_latencies(all_rows, "label_total_ms"),
        "overall_forward_latency": summarize_latencies(all_rows, "forward_ms"),
        "overall_prep_latency": summarize_latencies(all_rows, "prep_ms"),
        "overall_event_generation_latency": summarize_latencies([r for r in all_rows if r.get("event_emitted")], "event_generate_ms"),
        "audio_only_outputs": audio_outputs,
        "notes": [
            "Pseudo-realtime uses chronological VAD ticks and audio prefixes only; no future transcript is provided.",
            "label_total_ms includes audio preprocessing/tokenization/move plus one Thinker forward.",
            "event_generate_ms is measured only when BACKCHANNEL/SUPPORT transition is emitted.",
            "Talker audio is mixed into WAV outputs after the pseudo-realtime decisions; Talker synthesis is not counted as a recurring 0.5s tick operation.",
        ],
    }
    write_csv(OUT_DIR / "pseudorealtime_ticks_all.csv", all_rows)
    write_json(OUT_DIR / "summary.json", overall)
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
