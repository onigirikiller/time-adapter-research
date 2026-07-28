from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/omni3b_single_token_lora_vad_realtime_demo_v2"
CACHE_DIR = ROOT / ".cache/huggingface"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
QWEN3TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
BEST_LORA = ROOT / "artifacts/omni3b_single_token_lora_v1/clean0p6b_large_slash_l3_a4_audio_only_1000pc_e2_fulltest/best_lora"
LAYER = 3
ALPHA = 4.0
POSITION = "all_tokens"
CODEBOOK = {"WAIT": "/W", "BACKCHANNEL": "/B", "SUPPORT": "/S"}
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
VAD_FRAME_S = 0.05
VAD_HOP_S = 0.02


SPECS = {
    "midpause_hesitant_disclosure": {
        "title": "Mid-pause hesitant disclosure",
        "profile": "hesitant",
        "chunks": [
            "Today at school",
            "um",
            "I do not really want to say it, but",
            "something happened and I have been thinking about it all day",
        ],
        "silences_after_chunks": [0.85, 1.05, 2.30, 6.0],
        "input_speaker": "serena",
        "language": "English",
        "instruct": "Speak naturally and hesitantly, as one side of a private conversation. No extra commentary.",
        "backchannel_instruction": "Give a very short acknowledgement only, such as yeah or mm-hm.",
        "support_instruction": "Give one short supportive sentence. Do not mention labels, timing, or system details.",
    },
    "self_repair_many_pauses": {
        "title": "Self repair with multiple pauses",
        "profile": "self_repair",
        "chunks": [
            "I went to the office",
            "no, wait",
            "I mean I went to the counselor's room",
            "and then I forgot what I wanted to ask",
        ],
        "silences_after_chunks": [0.60, 0.95, 1.20, 5.0],
        "input_speaker": "serena",
        "language": "English",
        "instruct": "Speak naturally, with small self-corrections. No extra commentary.",
        "backchannel_instruction": "Give a very short acknowledgement without taking the turn.",
        "support_instruction": "Give one short supportive response only if it is clearly time to help.",
    },
    "asked_wait_long_silence": {
        "title": "User asks for time",
        "profile": "asked_wait",
        "chunks": [
            "Give me a second",
            "I am trying to find the right words",
            "please do not answer yet",
        ],
        "silences_after_chunks": [1.0, 2.0, 6.0],
        "input_speaker": "serena",
        "language": "English",
        "instruct": "Speak calmly and naturally. No extra commentary.",
        "backchannel_instruction": "Give only a quiet acknowledgement without taking the turn.",
        "support_instruction": "Respect that the user asked for time. If responding, keep it extremely brief.",
    },
    "finished_then_silence": {
        "title": "Finished thought then silence",
        "profile": "finished",
        "chunks": [
            "That is all I wanted to explain about what happened today",
        ],
        "silences_after_chunks": [6.0],
        "input_speaker": "serena",
        "language": "English",
        "instruct": "Speak clearly and naturally. No extra commentary.",
        "backchannel_instruction": "Give a very short acknowledgement.",
        "support_instruction": "Give a brief helpful response to continue the conversation.",
    },
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


single = import_module(ROOT / "scripts/run_omni3b_single_token_lora_v1.py", "single_token_vad_demo_core")
v4 = import_module(ROOT / "scripts/run_omni_talker_vad_threshold_demo_v4.py", "vad_v4_talker_core")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.reshape(-1), int(sr)


def resample_to_24k(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == 24000:
        return wav.astype(np.float32), sr
    x_old = np.linspace(0, 1, len(wav), endpoint=False)
    x_new = np.linspace(0, 1, int(len(wav) * 24000 / sr), endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32), 24000


def load_qwen3tts():
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Qwen3TTSModel.from_pretrained(
        QWEN3TTS_MODEL,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation="eager",
    )


def synth_chunk(tts, text: str, spec: dict, out_path: Path, force: bool):
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and out_path.stat().st_size > 4096 and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavs, sr = tts.generate_custom_voice(
        text=[text],
        language=[spec["language"]],
        speaker=[spec["input_speaker"]],
        instruct=[spec["instruct"]],
        max_new_tokens=360,
    )
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    wav = v4.fade_and_limit(v4.trim_audio(wav, int(sr), threshold=0.008), int(sr), max_s=12.0)
    sf.write(str(out_path), wav, int(sr))
    meta = {
        "path": str(out_path),
        "text": text,
        "model_id": QWEN3TTS_MODEL,
        "speaker": spec["input_speaker"],
        "sample_rate": int(sr),
        "duration_s": float(len(wav) / max(int(sr), 1)),
    }
    write_json(meta_path, meta)
    return meta


def build_input_stream(tts, key: str, spec: dict, out_dir: Path, force: bool):
    chunk_metas = []
    pieces = []
    sr_out = 24000
    timeline = []
    cursor = 0.0
    for i, text in enumerate(spec["chunks"]):
        chunk_path = out_dir / "chunks" / f"chunk_{i:02d}.wav"
        meta = synth_chunk(tts, text, spec, chunk_path, force)
        wav, sr = read_audio(Path(meta["path"]))
        wav, sr = resample_to_24k(wav, sr)
        pieces.append(wav)
        start = cursor
        cursor += len(wav) / sr_out
        silence_s = float(spec["silences_after_chunks"][i])
        timeline.append(
            {
                "chunk_index": i,
                "text": text,
                "speech_start_s": start,
                "speech_end_s": cursor,
                "silence_after_s": silence_s,
                "silence_start_s": cursor,
                "silence_end_s": cursor + silence_s,
            }
        )
        pieces.append(np.zeros(int(round(silence_s * sr_out)), dtype=np.float32))
        cursor += silence_s
        chunk_metas.append(meta)
    stream = np.concatenate(pieces).astype(np.float32)
    peak = float(np.max(np.abs(stream))) if stream.size else 0.0
    if peak > 0.98:
        stream = stream / peak * 0.98
    out_path = out_dir / "input_qwen3tts_mid_silence.wav"
    sf.write(str(out_path), stream, sr_out)
    meta = {
        "path": str(out_path),
        "sample_rate": sr_out,
        "duration_s": float(len(stream) / sr_out),
        "qwen3tts_model": QWEN3TTS_MODEL,
        "method": "Qwen3TTS chunks concatenated with zero-valued PCM silence. Silence is not requested in TTS prompt.",
        "timeline": timeline,
        "chunk_metas": chunk_metas,
    }
    write_json(out_path.with_suffix(".json"), meta)
    return out_path, stream, sr_out, meta


def build_vad_rows(stream: np.ndarray, sr: int, spec: dict, threshold: float, tick_s: float):
    frame = max(1, int(round(VAD_FRAME_S * sr)))
    hop = max(1, int(round(VAD_HOP_S * sr)))
    rows, frames = [], []
    last_voice_time = 0.0
    last_tick_time = -999.0
    was_below = False
    utterance_start_time = None
    previous_tick_time = None
    prev_asr_marker = 0.0
    seen_voice = False
    for start in range(0, max(1, len(stream) - frame + 1), hop):
        end = min(len(stream), start + frame)
        frame_end_s = round(end / sr, 4)
        seg = stream[start:end]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        vad_speaking = bool(rms >= threshold)
        if vad_speaking:
            seen_voice = True
            last_voice_time = frame_end_s
            if utterance_start_time is None:
                utterance_start_time = round(start / sr, 4)
            was_below = False
            prev_asr_marker = frame_end_s
            frames.append({"time_s": frame_end_s, "rms": rms, "vad_speaking": True, "silence_elapsed": 0.0})
            continue
        if not seen_voice:
            frames.append({"time_s": frame_end_s, "rms": rms, "vad_speaking": False, "silence_elapsed": 0.0})
            continue
        silence_elapsed = round(max(0.0, frame_end_s - last_voice_time), 4)
        frames.append({"time_s": frame_end_s, "rms": rms, "vad_speaking": False, "silence_elapsed": silence_elapsed})
        should_tick = (not was_below) or (frame_end_s - last_tick_time >= tick_s - 1e-9)
        was_below = True
        if not should_tick:
            continue
        delta_t = 0.0 if previous_tick_time is None else round(frame_end_s - previous_tick_time, 4)
        asr_changed = previous_tick_time is None or prev_asr_marker > previous_tick_time
        previous_tick_time = frame_end_s
        last_tick_time = frame_end_s
        utterance_elapsed = frame_end_s if utterance_start_time is None else round(frame_end_s - utterance_start_time, 4)
        rows.append(
            {
                "id": f"{spec['profile']}_vad_tick_{len(rows):03d}",
                "context_id": f"{spec['profile']}_vad_demo",
                "clock_s": float(frame_end_s),
                "timepoint_s": float(silence_elapsed),
                "silence_seconds": float(silence_elapsed),
                "features": {
                    "silence_elapsed": float(silence_elapsed),
                    "delta_t": float(delta_t),
                    "utterance_elapsed": float(utterance_elapsed),
                    "is_user_speaking": False,
                    "asr_changed": bool(asr_changed),
                },
                "profile": spec["profile"],
                "fragment": "[audio-only; transcript withheld from Omni]",
                "label": "",
                "vad_rms": rms,
                "vad_threshold": threshold,
                "vad_trigger": "rms_below_threshold",
                "last_voice_time_s": float(last_voice_time),
            }
        )
    vad_meta = {
        "vad_type": "fixed_rms_threshold_realtime_scan",
        "rms_threshold": threshold,
        "frame_s": VAD_FRAME_S,
        "hop_s": VAD_HOP_S,
        "tick_interval_while_below_threshold_s": tick_s,
        "frames": frames,
        "note": "Rows are emitted while scanning frames in chronological order. No final speech end oracle is used.",
    }
    return rows, vad_meta


def write_segments(stream: np.ndarray, sr: int, rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in rows:
        end = max(1, min(len(stream), int(round(float(row["clock_s"]) * sr))))
        segment = stream[:end].astype(np.float32)
        path = out_dir / f"vad_tick_{int(round(float(row['clock_s']) * 1000)):06d}ms.wav"
        sf.write(str(path), segment, sr)
        out.append({**row, "audio_path": str(path), "segment_audio_path": str(path), "segment_duration_s": float(len(segment) / sr)})
    return out


def load_classifier():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.disable_talker()
    model.config.use_cache = False
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model = single.attach_existing_lora(model, BEST_LORA)
    model.eval()
    token_ids = single.label_token_ids(processor.tokenizer, CODEBOOK)
    adapter, _ = single.load_time_adapter(LAYER)
    return processor, model, dtype, token_ids, adapter


def classify_rows(processor, model, dtype, token_ids, adapter, rows: list[dict]):
    vectors = single.adapter_predict(adapter, rows)
    decisions = []
    with torch.inference_mode():
        for row, vector in zip(rows, vectors):
            inputs, _, _ = single.prepare_inputs(
                processor,
                row,
                "audio_only",
                CODEBOOK,
                label=None,
                audio_timing_mode="row_audio",
            )
            moved = single.move_inputs(inputs, model.device, dtype)
            hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
            handle = single.hook_module(model, LAYER).register_forward_hook(hook)
            try:
                outputs = model.thinker(**moved, use_audio_in_video=False)
            finally:
                handle.remove()
            logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
            label_logits = np.array([logits[token_ids[label]] for label in LABELS], dtype=np.float64)
            probs = single.softmax3(label_logits)
            raw_label = LABELS[int(np.argmax(probs))]
            stat = hook.stats[0] if hook.stats else {}
            decisions.append(
                {
                    **row,
                    "raw_label": raw_label,
                    "emitted_action": raw_label,
                    "p_WAIT": float(probs[0]),
                    "p_BACKCHANNEL": float(probs[1]),
                    "p_SUPPORT": float(probs[2]),
                    "time_vector_norm": stat.get("injected_norm", 0.0),
                    "context_norm": stat.get("hidden_norm", 0.0),
                    "time_context_cosine": stat.get("hidden_injected_cosine", 0.0),
                    "hook_calls": hook.calls,
                    "layer": LAYER,
                    "alpha": ALPHA,
                }
            )
    prev = np.zeros(2048, dtype=np.float32)
    for d, vec in zip(decisions, vectors):
        d["time_vector_step_l2"] = float(np.linalg.norm(vec - prev))
        prev = vec
    return decisions


def load_talker():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model.eval()
    return model, processor


def select_speech_events(decisions: list[dict]):
    events = []
    last = "WAIT"
    for d in decisions:
        label = d["raw_label"]
        if label == "WAIT":
            last = "WAIT"
            continue
        if label != last:
            events.append(d)
        last = label
    return events


def generate_response_events(model, processor, spec: dict, speech_decisions: list[dict], out_dir: Path, speaker: str, force: bool):
    events = []
    for i, decision in enumerate(speech_decisions):
        action = decision["raw_label"]
        out_path = out_dir / "responses" / f"{i:02d}_{int(round(decision['clock_s'] * 1000)):06d}ms_{action.lower()}_{speaker.lower()}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = v4.generate_omni_talker_response(
            model,
            processor,
            spec,
            action,
            out_path,
            speaker,
            Path(decision["segment_audio_path"]),
            force=force,
        )
        event = {
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
        events.append(event)
    return events


def mix_audio(stream: np.ndarray, sr: int, events: list[dict], out_path: Path):
    track = stream.astype(np.float32).copy()
    for event in events:
        wav, wav_sr = read_audio(Path(event["audio_path"]))
        wav, _ = resample_to_24k(wav, wav_sr)
        start = int(round(float(event["clock_s"]) * sr))
        needed = start + len(wav)
        if needed > len(track):
            track = np.pad(track, (0, needed - len(track)), mode="constant")
        end = min(len(track), needed)
        if end > start:
            track[start:end] = np.clip(track[start:end] + wav[: end - start] * 0.92, -0.98, 0.98)
        event["actual_audio_end_s"] = float(end / sr)
        event["overlaps_future_user_speech"] = False
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track = track / peak * 0.98
    sf.write(str(out_path), track, sr)
    return {"path": str(out_path), "duration_s": float(len(track) / sr), "sample_rate": sr}


def get_font(size: int):
    for path in ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/arial.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(draw, text: str, font, width: int):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if draw.textbbox((0, 0), trial, font=font)[2] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_bar(draw, x, y, width, label, prob, color, font):
    draw.text((x, y), label, fill=(235, 238, 244), font=font)
    bx = x + 170
    draw.rectangle((bx, y + 4, bx + width, y + 24), fill=(45, 51, 62), outline=(100, 108, 122))
    draw.rectangle((bx, y + 4, bx + int(width * prob), y + 24), fill=color)
    draw.text((bx + width + 10, y), f"{prob:.2f}", fill=(235, 238, 244), font=font)


def render_video(spec: dict, stream_meta: dict, decisions: list[dict], vad_meta: dict, events: list[dict], mixed_wav: Path, out_mp4: Path):
    fps, w, h = 10, 1280, 720
    duration = float(stream_meta["duration_s"])
    noaudio = out_mp4.with_name(out_mp4.stem + "_noaudio.mp4")
    title_font, font, small = get_font(30), get_font(23), get_font(17)
    colors = {"WAIT": (102, 194, 165), "BACKCHANNEL": (252, 141, 98), "SUPPORT": (141, 160, 203)}
    frames = vad_meta["frames"]
    threshold = float(vad_meta["rms_threshold"])
    max_rms = max([float(f["rms"]) for f in frames] + [threshold * 2.0, 1e-6])
    writer = imageio.get_writer(str(noaudio), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    for frame_i in range(int(math.ceil(duration * fps))):
        now = frame_i / fps
        current_decision = None
        for d in decisions:
            if float(d["clock_s"]) <= now + 1e-6:
                current_decision = d
            else:
                break
        if current_decision is None:
            current_decision = decisions[0] if decisions else {}
        current_vad = None
        for f in frames:
            if float(f["time_s"]) <= now + 1e-6:
                current_vad = f
            else:
                break
        current_event = next((e for e in events if e["clock_s"] <= now <= e.get("actual_audio_end_s", e["clock_s"])), None)
        img = Image.new("RGB", (w, h), (18, 22, 30))
        draw = ImageDraw.Draw(img)
        draw.text((42, 30), stream_meta.get("demo_title", "DirectLM VAD-Threshold Realtime Demo"), fill=(248, 249, 251), font=title_font)
        draw.text((44, 78), spec["title"], fill=(180, 198, 255), font=font)
        draw.text((44, 112), stream_meta.get("input_description", "Qwen3TTS chunks + zero PCM mid-silences. Omni classifier receives audio segment only."), fill=(188, 196, 210), font=small)
        y = 150
        viewer_text = stream_meta.get("viewer_text")
        if not viewer_text:
            viewer_text = "Viewer transcript only: " + " ... ".join(spec.get("chunks", []))
        for line in wrap_text(draw, viewer_text, small, 780)[:4]:
            draw.text((44, y), line, fill=(235, 238, 244), font=small)
            y += 24
        draw.text((860, 130), f"clock {now:05.2f}s", fill=(248, 249, 251), font=font)
        vad_label = "-"
        rms = 0.0
        silence_elapsed = 0.0
        if current_vad:
            rms = float(current_vad["rms"])
            vad_label = "SPEECH" if current_vad["vad_speaking"] else "BELOW THRESHOLD"
            silence_elapsed = float(current_vad["silence_elapsed"])
        draw.text((860, 164), f"VAD: {vad_label}", fill=(255, 213, 128), font=small)
        draw.text((860, 190), f"RMS {rms:.4f} / threshold {threshold:.4f}", fill=(255, 213, 128), font=small)
        draw.text((860, 216), f"silence_elapsed {silence_elapsed:.2f}s", fill=(255, 213, 128), font=small)
        if current_event:
            draw.text((860, 246), f"Omni Talker: {current_event['action']}", fill=(255, 242, 174), font=font)
            for line in wrap_text(draw, current_event["assistant_text"], small, 360)[:2]:
                draw.text((860, 282), line, fill=(255, 242, 174), font=small)
        y = 310
        if current_decision:
            draw.text((44, y), f"Current DirectLM label: {current_decision['raw_label']} @ {current_decision['clock_s']:.2f}s", fill=(248, 249, 251), font=font)
            y += 42
            draw_bar(draw, 44, y, 390, "WAIT", float(current_decision["p_WAIT"]), colors["WAIT"], small)
            y += 40
            draw_bar(draw, 44, y, 390, "BACKCHANNEL", float(current_decision["p_BACKCHANNEL"]), colors["BACKCHANNEL"], small)
            y += 40
            draw_bar(draw, 44, y, 390, "SUPPORT", float(current_decision["p_SUPPORT"]), colors["SUPPORT"], small)
            y += 48
            draw.text((44, y), f"tick source: RMS below threshold, segment duration {current_decision['segment_duration_s']:.2f}s", fill=(188, 196, 210), font=small)
            y += 25
            draw.text((44, y), f"time_vector_norm {current_decision['time_vector_norm']:.2f}, hook_calls {current_decision['hook_calls']}", fill=(188, 196, 210), font=small)
        # RMS graph
        gx0, gy0, gx1, gy1 = 44, 520, 1236, 575
        draw.rectangle((gx0, gy0, gx1, gy1), fill=(28, 34, 45), outline=(83, 92, 110))
        thresh_y = gy1 - int(min(1.0, threshold / max_rms) * (gy1 - gy0))
        draw.line((gx0, thresh_y, gx1, thresh_y), fill=(255, 110, 110), width=2)
        prev = None
        for f in frames:
            t = float(f["time_s"])
            if t > now:
                break
            x = gx0 + int((t / duration) * (gx1 - gx0))
            yv = gy1 - int(min(1.0, float(f["rms"]) / max_rms) * (gy1 - gy0))
            if prev:
                draw.line((prev[0], prev[1], x, yv), fill=(255, 213, 128), width=2)
            prev = (x, yv)
        draw.text((44, 492), "Realtime VAD RMS scan", fill=(141, 211, 199), font=small)
        # timeline
        tx0, ty0, tx1, ty1 = 44, 615, 1236, 646
        draw.rectangle((tx0, ty0, tx1, ty1), fill=(45, 51, 62), outline=(110, 119, 134))
        for item in stream_meta["timeline"]:
            sx0 = tx0 + int(float(item["speech_start_s"]) / duration * (tx1 - tx0))
            sx1 = tx0 + int(float(item["speech_end_s"]) / duration * (tx1 - tx0))
            draw.rectangle((sx0, ty0, sx1, ty1), fill=(65, 145, 170))
        for d in decisions:
            x = tx0 + int(float(d["clock_s"]) / duration * (tx1 - tx0))
            col = colors.get(d["raw_label"], (255, 255, 255))
            draw.line((x, ty0 - 12, x, ty1 + 12), fill=col, width=2)
        for e in events:
            x = tx0 + int(float(e["clock_s"]) / duration * (tx1 - tx0))
            draw.text((x - 24, ty0 - 38), e["action"][:4], fill=(255, 242, 174), font=small)
        nx = tx0 + int(now / duration * (tx1 - tx0))
        draw.line((nx, ty0 - 28, nx, ty1 + 28), fill=(255, 255, 255), width=3)
        draw.text((44, 658), stream_meta.get("timeline_legend", "blue=speech chunks, gaps=zero PCM silence, vertical lines=VAD-triggered DirectLM ticks"), fill=(188, 196, 210), font=small)
        writer.append_data(np.asarray(img))
    writer.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(noaudio), "-i", str(mixed_wav), "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(out_mp4)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return out_mp4


def process_specs(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = SPECS if args.demo == "all" else {args.demo: SPECS[args.demo]}
    prepared = {}
    tts = load_qwen3tts()
    for key, spec in specs.items():
        out_dir = OUT_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        input_path, stream, sr, stream_meta = build_input_stream(tts, key, spec, out_dir, args.force_input)
        rows, vad_meta = build_vad_rows(stream, sr, spec, args.vad_threshold, args.tick)
        rows = write_segments(stream, sr, rows, out_dir / "segments")
        prepared[key] = {"spec": spec, "out_dir": out_dir, "input_path": input_path, "stream": stream, "sr": sr, "stream_meta": stream_meta, "rows": rows, "vad_meta": vad_meta}
        write_json(out_dir / "vad_rows.json", {"rows": rows, "vad": {k: v for k, v in vad_meta.items() if k != "frames"}})
    del tts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    processor, classifier, dtype, token_ids, adapter = load_classifier()
    for key, item in prepared.items():
        decisions = classify_rows(processor, classifier, dtype, token_ids, adapter, item["rows"])
        item["decisions"] = decisions
        write_csv(item["out_dir"] / "per_tick_decisions.csv", decisions)
    del classifier, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    talker_model, talker_processor = load_talker()
    summaries = {}
    for key, item in prepared.items():
        speech_decisions = select_speech_events(item["decisions"])
        if args.max_events > 0:
            speech_decisions = speech_decisions[: args.max_events]
        events = generate_response_events(talker_model, talker_processor, item["spec"], speech_decisions, item["out_dir"], args.speaker, args.force_talker)
        audio_meta = mix_audio(item["stream"], item["sr"], events, item["out_dir"] / "demo_mix.wav")
        video = render_video(item["spec"], item["stream_meta"], item["decisions"], item["vad_meta"], events, Path(audio_meta["path"]), item["out_dir"] / "demo_vad_realtime_directlm.mp4")
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
                "base": MODEL_ID,
                "lora": str(BEST_LORA),
                "codebook": CODEBOOK,
                "layer": LAYER,
                "alpha": ALPHA,
                "prompt_mode": "audio_only",
                "classifier_input": "audio segment from stream start to current VAD tick only",
            },
        }
        write_json(item["out_dir"] / "summary.json", summary)
        summaries[key] = {"video": str(video), "events": events, "num_ticks": len(item["decisions"])}
        print(f"Wrote {video}", flush=True)
    write_json(OUT_DIR / "summary.json", summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=[*SPECS.keys(), "all"], default="all")
    parser.add_argument("--speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--vad-threshold", type=float, default=0.01)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--force-input", action="store_true")
    parser.add_argument("--force-talker", action="store_true")
    args = parser.parse_args()
    process_specs(args)


if __name__ == "__main__":
    main()
