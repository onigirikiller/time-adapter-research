from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/omni3b_delayed_backchannel_v2_qwen3tts_clean_0p6b/test.jsonl"
OUT_ROOT = ROOT / "artifacts/omni3b_realtime_webui_v1/benchmarks"
DEFAULT_PROFILES = ["asked_wait", "self_repair", "hesitant", "vulnerable", "finished", "direct_question"]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def load_contexts() -> list[dict]:
    contexts: dict[str, dict] = {}
    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            contexts.setdefault(row["context_id"], row)
    return list(contexts.values())


def choose_cases(count: int, seed: int, profiles: list[str]) -> list[dict]:
    rng = random.Random(seed)
    by_profile: dict[str, list[dict]] = {}
    for row in load_contexts():
        by_profile.setdefault(row["profile"], []).append(row)
    chosen = []
    for profile in profiles:
        candidates = by_profile.get(profile, [])
        if candidates:
            chosen.append(rng.choice(candidates))
        if len(chosen) >= count:
            return chosen
    remaining = [row for rows in by_profile.values() for row in rows if row not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen


def pcm16_bytes(chunk: np.ndarray) -> bytes:
    clipped = np.clip(chunk, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype("<i2").tobytes()


async def run_case(args, case: dict, case_dir: Path) -> dict:
    audio_path = (ROOT / Path(case["audio_path"])).resolve()
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    chunk_samples = max(1, int(round(sample_rate * args.chunk_ms / 1000.0)))
    trailing = np.zeros(int(round(args.trailing_silence * sample_rate)), dtype=np.float32)
    stream = np.concatenate([audio, trailing]).astype(np.float32)
    messages: list[dict] = []
    started = time.perf_counter()

    async with websockets.connect(args.ws_url, max_size=None, ping_interval=20, ping_timeout=90) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "sampleRate": sample_rate,
                    "vadThreshold": args.vad_threshold,
                    "tickSeconds": args.tick_seconds,
                    "maxContextSeconds": args.max_context_seconds,
                }
            )
        )

        async def receive_messages():
            async for raw in ws:
                item = json.loads(raw)
                item["client_received_wall_s"] = time.perf_counter() - started
                messages.append(item)
                if item.get("type") == "session_stopped":
                    return

        receiver = asyncio.create_task(receive_messages())
        next_send = time.perf_counter()
        for offset in range(0, len(stream), chunk_samples):
            chunk = stream[offset : offset + chunk_samples]
            await ws.send(pcm16_bytes(chunk))
            if args.realtime:
                next_send += len(chunk) / sample_rate
                await asyncio.sleep(max(0.0, next_send - time.perf_counter()))
        stream_finished_at = time.perf_counter()
        await asyncio.sleep(args.settle_seconds)
        wait_deadline = time.perf_counter() + args.max_response_wait
        while time.perf_counter() < wait_deadline:
            event_times = [item.get("client_received_wall_s", 0.0) for item in messages if item.get("type") == "event_detected"]
            audio_times = [
                item.get("client_received_wall_s", 0.0)
                for item in messages
                if item.get("type") == "assistant_audio_ready"
            ]
            error_times = [
                item.get("client_received_wall_s", 0.0)
                for item in messages
                if item.get("type") == "generation_status" and item.get("stage") == "generation_error"
            ]
            latest_event_has_audio = bool(event_times and audio_times and max(audio_times) >= max(event_times))
            latest_event_failed = bool(event_times and error_times and max(error_times) >= max(event_times))
            if (
                latest_event_has_audio
                or latest_event_failed
                or (not event_times and time.perf_counter() - stream_finished_at >= args.idle_wait)
            ):
                break
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(receiver, timeout=5.0)
        except asyncio.TimeoutError:
            receiver.cancel()

    ticks = [m for m in messages if m.get("type") == "tick"]
    events = [m for m in messages if m.get("type") == "event_detected"]
    statuses = [m for m in messages if m.get("type") == "generation_status"]
    audio_ready = [m for m in messages if m.get("type") == "assistant_audio_ready"]
    label_latencies = [float(m.get("latency_ms", {}).get("server_wall", 0.0)) for m in ticks]
    tick_walls = [float(m["client_received_wall_s"]) for m in ticks]
    tick_intervals = [b - a for a, b in zip(tick_walls, tick_walls[1:])]
    summary = {
        "context_id": case["context_id"],
        "profile": case["profile"],
        "fragment_for_audit_only_not_sent": case["fragment"],
        "qwen3tts_model_id": case.get("qwen3tts_model_id"),
        "audio_path": str(audio_path),
        "speech_duration_s": len(audio) / sample_rate,
        "trailing_zero_pcm_s": args.trailing_silence,
        "stream_wall_s": time.perf_counter() - started,
        "tick_count": len(ticks),
        "event_count": len(events),
        "audio_response_count": len(audio_ready),
        "labels": [m.get("label") for m in ticks],
        "silence_elapsed": [m.get("silence_elapsed") for m in ticks],
        "label_latency_ms": {
            "mean": statistics.fmean(label_latencies) if label_latencies else 0.0,
            "p50": percentile(label_latencies, 50),
            "p95": percentile(label_latencies, 95),
            "max": max(label_latencies, default=0.0),
        },
        "tick_receive_interval_s": {
            "mean": statistics.fmean(tick_intervals) if tick_intervals else 0.0,
            "p95": percentile(tick_intervals, 95),
            "max": max(tick_intervals, default=0.0),
        },
        "generated_responses": [m.get("generated_response") for m in statuses if m.get("stage") == "text_ready"],
        "generation_stages": [m.get("stage") for m in statuses],
        "response_audio_urls": [m.get("audio_url") for m in audio_ready],
        "decision_model_modes": sorted({m.get("decision_model_mode") for m in ticks if m.get("decision_model_mode")}),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "messages.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages),
        encoding="utf-8",
    )
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def async_main(args):
    run_dir = OUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cases = choose_cases(args.cases, args.seed, args.profiles)
    summaries = []
    for index, case in enumerate(cases):
        print(f"[{index + 1}/{len(cases)}] {case['profile']} {case['context_id']}", flush=True)
        summary = await run_case(args, case, run_dir / f"{index:02d}_{case['profile']}_{case['context_id']}")
        summaries.append(summary)
        print(
            f"  ticks={summary['tick_count']} labels={summary['labels']} "
            f"label_p50={summary['label_latency_ms']['p50']:.1f}ms responses={summary['generated_responses']}",
            flush=True,
        )
    aggregate = {
        "run_dir": str(run_dir),
        "config": vars(args),
        "cases": summaries,
        "mean_label_latency_ms": statistics.fmean(
            [s["label_latency_ms"]["mean"] for s in summaries if s["tick_count"]]
        )
        if summaries
        else 0.0,
        "total_ticks": sum(s["tick_count"] for s in summaries),
        "total_events": sum(s["event_count"] for s in summaries),
        "total_audio_responses": sum(s["audio_response_count"] for s in summaries),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="ws://127.0.0.1:7865/ws")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--chunk-ms", type=float, default=42.6667)
    parser.add_argument("--trailing-silence", type=float, default=9.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--idle-wait", type=float, default=2.0)
    parser.add_argument("--max-response-wait", type=float, default=90.0)
    parser.add_argument("--vad-threshold", type=float, default=0.01)
    parser.add_argument("--tick-seconds", type=float, default=0.5)
    parser.add_argument("--max-context-seconds", type=float, default=24.0)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
