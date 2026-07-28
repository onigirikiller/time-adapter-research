from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/omni3b_control_response_lora_demo_v1"
RESPONSE_LORA = ROOT / "artifacts/omni3b_control_response_lora_v1/control_response_3000_e2_from_label_lora/best_lora"
REAL_AUDIO_DIR = ROOT / "data/drive_test_audio_2026-06-28"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
CACHE_DIR = ROOT / ".cache/huggingface"
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


single = import_module(ROOT / "scripts/run_omni3b_single_token_lora_v1.py", "single_token_core_response_demo")
single.DATA_DIR = ROOT / "data/omni3b_delayed_backchannel_v2_qwen3tts_clean_0p6b"
single.SOURCE_DATA_DIR = ROOT / "data/omni3b_sequential_v2"
v2 = import_module(ROOT / "scripts/run_single_token_lora_vad_realtime_demo_v2.py", "single_token_vad_response_demo")
drive = import_module(ROOT / "scripts/run_single_token_lora_drive_audio_vad_demo_v1.py", "drive_vad_response_demo")
resp = import_module(ROOT / "scripts/run_omni3b_control_response_lora_v1.py", "response_lora_core_demo")


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


def load_response_thinker():
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
    if hasattr(model.thinker, "config"):
        model.thinker.config.use_cache = False
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model = single.attach_existing_lora(model, RESPONSE_LORA)
    model.eval()
    token_ids = single.label_token_ids(processor.tokenizer, CODEBOOK)
    adapter, _ = single.load_time_adapter(LAYER)
    return processor, model, dtype, token_ids, adapter


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


def parse_generated(text: str) -> tuple[str, str]:
    return resp.parse_control_text(text)


def build_row(case: dict, tick_row: dict):
    return {
        **tick_row,
        "profile": case["profile"],
        "fragment": "[audio-only; transcript withheld from Omni]",
        "label": "",
    }


def classify_and_generate(processor, model, dtype, token_ids, adapter, rows: list[dict], case: dict):
    vectors = single.adapter_predict(adapter, rows)
    out = []
    with torch.inference_mode():
        for i, (row, vector) in enumerate(zip(rows, vectors)):
            prompt_inputs, prompt_len, _ = single.prepare_inputs(
                processor,
                row,
                "audio_only",
                CODEBOOK,
                label=None,
                audio_timing_mode="row_audio",
            )
            moved = single.move_inputs(prompt_inputs, model.device, dtype)
            hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
            handle = single.hook_module(model, LAYER).register_forward_hook(hook)
            try:
                outputs = model.thinker(**moved, use_audio_in_video=False)
                logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
            finally:
                handle.remove()
            label_logits = np.array([logits[token_ids[label]] for label in LABELS], dtype=np.float64)
            probs = single.softmax3(label_logits)
            raw_label = LABELS[int(np.argmax(probs))]

            gen_hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
            gen_handle = single.hook_module(model, LAYER).register_forward_hook(gen_hook)
            try:
                generated = model.thinker.generate(
                    **moved,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=False,
                    use_audio_in_video=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            finally:
                gen_handle.remove()
            generated_text = processor.tokenizer.decode(
                generated[0, prompt_len:].detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_label, generated_response = parse_generated(generated_text)
            stat = hook.stats[0] if hook.stats else {}
            out.append(
                {
                    **row,
                    "raw_label": raw_label,
                    "generated_label": generated_label,
                    "generated_response": generated_response,
                    "generated_raw": generated_text,
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
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"{case['key']} generated {i + 1}/{len(rows)}", flush=True)
    return out


def select_events(decisions: list[dict]):
    events = []
    last = "WAIT"
    for row in decisions:
        label = row["generated_label"] if row["generated_label"] in {"WAIT", "BACKCHANNEL", "SUPPORT"} else row["raw_label"]
        if label == "WAIT":
            last = "WAIT"
            continue
        if label != last:
            event = {**row, "raw_label": label}
            events.append(event)
        last = label
    return events


def exact_talker_prompt(text: str) -> str:
    cleaned = text.strip()
    return (
        "Speak exactly the following listener response aloud, with no extra words.\n"
        f"Response: {cleaned}\n"
        "Return only that response text."
    )


def generate_exact_talker_audio(model, processor, response_text: str, out_path: Path, speaker: str, force: bool = False):
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and out_path.stat().st_size > 4096 and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    conv = [
        {"role": "system", "content": [{"type": "text", "text": v2.v4.DEFAULT_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": exact_talker_prompt(response_text)}]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = v2.v4.move_inputs(inputs, model.device, model.dtype)
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            return_audio=True,
            speaker=speaker,
            thinker_max_new_tokens=32,
            talker_max_new_tokens=768,
            talker_do_sample=True,
            talker_top_k=40,
            talker_top_p=0.8,
            talker_temperature=0.85,
            talker_repetition_penalty=1.05,
            use_audio_in_video=False,
        )
    text_ids, audio = result
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    spoken_text = decoded.split("assistant\n")[-1].strip()
    wav = audio.detach().float().cpu().numpy().reshape(-1)
    max_s = 2.0 if len(response_text.split()) <= 4 else 5.0
    wav = v2.v4.fade_and_limit(v2.v4.trim_audio(wav, 24000, threshold=0.006), 24000, max_s=max_s)
    metrics = v2.v4.audio_metrics(wav, 24000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, 24000)
    meta = {
        "requested_text": response_text,
        "spoken_text": spoken_text,
        "path": str(out_path),
        "speaker": speaker,
        "sample_rate": 24000,
        "metrics": metrics,
        "source": "Qwen2.5-Omni-3B Talker exact-text prompt",
    }
    write_json(meta_path, meta)
    return meta


def build_case_inputs(args):
    cases = []
    # Real WAITGOAL as provided.
    wav, sr = v2.read_audio(REAL_AUDIO_DIR / "WAITGOAL.wav")
    wav, sr = v2.resample_to_24k(wav, sr)
    cases.append(
        {
            "key": "real_WAITGOAL",
            "title": "Real WAITGOAL, original duration",
            "profile": "asked_wait",
            "stream": wav,
            "sr": sr,
            "viewer_text": "Real user audio WAITGOAL; transcript withheld from classifier.",
            "method": "User Drive WAV converted to mono 24 kHz. No extra silence.",
        }
    )
    # Real SUPPORTGOAL with enough trailing silence to expose SUPPORT.
    wav, sr = v2.read_audio(REAL_AUDIO_DIR / "SUPPORTGOAL.wav")
    wav, sr = v2.resample_to_24k(wav, sr)
    wav = np.concatenate([wav, np.zeros(int(round(args.real_support_extra_s * sr)), dtype=np.float32)]).astype(np.float32)
    cases.append(
        {
            "key": f"real_SUPPORTGOAL_plus_{int(args.real_support_extra_s)}s",
            "title": f"Real SUPPORTGOAL + {args.real_support_extra_s:g}s trailing silence",
            "profile": "vulnerable",
            "stream": wav,
            "sr": sr,
            "viewer_text": f"Real user audio SUPPORTGOAL + {args.real_support_extra_s:g}s zero PCM trailing silence; transcript withheld.",
            "method": "User Drive WAV converted to mono 24 kHz plus trailing zero-valued PCM silence.",
        }
    )
    # Reuse Qwen3TTS forced-silence inputs from the previous controlled demo.
    tts_base = ROOT / "artifacts/omni3b_single_token_lora_qwen3tts_forced_silence_vad_demo_v1"
    for key, title, profile in [
        ("wait_text_medium", "Qwen3TTS WAIT text, medium forced silence", "asked_wait"),
        ("support_text_medium", "Qwen3TTS SUPPORT text, medium forced silence", "vulnerable"),
    ]:
        path = tts_base / key / "input_qwen3tts_forced_zero_pcm_silence.wav"
        wav, sr = v2.read_audio(path)
        cases.append(
            {
                "key": "tts_" + key,
                "title": title,
                "profile": profile,
                "stream": wav,
                "sr": sr,
                "viewer_text": f"Qwen3TTS forced-silence input {key}; transcript shown only to viewer, not classifier.",
                "method": "Qwen3TTS speech chunks plus forced zero-valued PCM silence from prior controlled demo.",
            }
        )
    return cases


def prepare_case(case: dict, args):
    out_dir = OUT_DIR / case["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = np.asarray(case["stream"], dtype=np.float32)
    peak = float(np.max(np.abs(stream))) if stream.size else 0.0
    if peak > 0.98:
        stream = stream / peak * 0.98
    input_path = out_dir / "input_audio_24k.wav"
    sf.write(str(input_path), stream, int(case["sr"]))
    spec = {
        "title": case["title"],
        "profile": case["profile"],
        "chunks": [],
        "backchannel_instruction": "",
        "support_instruction": "",
    }
    rows, vad_meta = v2.build_vad_rows(stream, int(case["sr"]), spec, args.vad_threshold, args.tick)
    rows = v2.write_segments(stream, int(case["sr"]), rows, out_dir / "segments")
    rows = [build_row(case, row) for row in rows]
    speech_intervals = drive.speech_intervals_from_vad(vad_meta["frames"], len(stream) / int(case["sr"]), float(vad_meta["frame_s"]), float(vad_meta["hop_s"]))
    stream_meta = {
        "path": str(input_path),
        "duration_s": float(len(stream) / int(case["sr"])),
        "sample_rate": int(case["sr"]),
        "method": case["method"],
        "demo_title": "Control+Response LoRA VAD Demo",
        "input_description": "New LoRA generates /W, /B text, or /S text. Fixed VAD; classifier receives audio prefix only.",
        "viewer_text": case["viewer_text"],
        "timeline": speech_intervals,
        "timeline_legend": "blue=VAD speech regions, gaps=below-threshold audio, vertical lines=VAD-triggered ticks",
    }
    return {
        **case,
        "out_dir": out_dir,
        "stream": stream,
        "input_path": input_path,
        "spec": spec,
        "rows": rows,
        "vad_meta": vad_meta,
        "stream_meta": stream_meta,
        "speech_intervals": speech_intervals,
    }


def generate_events(talker_model, talker_processor, decisions: list[dict], out_dir: Path, speaker: str, force: bool):
    events = []
    for i, row in enumerate(select_events(decisions)):
        response_text = row.get("generated_response", "").strip()
        if not response_text:
            continue
        action = row["raw_label"]
        out_path = out_dir / "responses" / f"{i:02d}_{int(round(float(row['clock_s']) * 1000)):06d}ms_{action.lower()}_{speaker.lower()}.wav"
        meta = generate_exact_talker_audio(talker_model, talker_processor, response_text, out_path, speaker, force=force)
        events.append(
            {
                "event_index": i,
                "action": action,
                "clock_s": float(row["clock_s"]),
                "silence_elapsed": float(row["features"]["silence_elapsed"]),
                "audio_path": meta["path"],
                "assistant_text": response_text,
                "talker_spoken_text": meta["spoken_text"],
                "speaker": speaker,
                "duration_s": meta["metrics"]["duration_s"],
                "source_segment": row["segment_audio_path"],
            }
        )
    return events


def process(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [prepare_case(case, args) for case in build_case_inputs(args)]
    processor, thinker, dtype, token_ids, adapter = load_response_thinker()
    for case in cases:
        decisions = classify_and_generate(processor, thinker, dtype, token_ids, adapter, case["rows"], case)
        case["decisions"] = decisions
        write_csv(case["out_dir"] / "per_tick_decisions.csv", decisions)
    del thinker, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    talker_model, talker_processor = load_talker()
    summaries = {}
    for case in cases:
        events = generate_events(talker_model, talker_processor, case["decisions"], case["out_dir"], args.speaker, args.force_talker)
        audio_meta = v2.mix_audio(case["stream"], int(case["sr"]), events, case["out_dir"] / "demo_mix.wav")
        drive.mark_future_overlap(events, case["speech_intervals"])
        render_meta = {**case["stream_meta"], "duration_s": max(float(case["stream_meta"]["duration_s"]), float(audio_meta["duration_s"]))}
        video = v2.render_video(case["spec"], render_meta, case["decisions"], case["vad_meta"], events, Path(audio_meta["path"]), case["out_dir"] / "demo_control_response_lora_vad.mp4")
        summary = {
            "input_audio": str(case["input_path"]),
            "stream_meta": case["stream_meta"],
            "vad": {k: v for k, v in case["vad_meta"].items() if k != "frames"},
            "decisions": case["decisions"],
            "events": events,
            "mixed_audio": audio_meta,
            "video": str(video),
            "model": {
                "base": MODEL_ID,
                "response_lora": str(RESPONSE_LORA),
                "codebook": CODEBOOK,
                "layer": LAYER,
                "alpha": ALPHA,
                "prompt_mode": "audio_only",
            },
        }
        write_json(case["out_dir"] / "summary.json", summary)
        summaries[case["key"]] = {"video": str(video), "events": events, "num_ticks": len(case["decisions"])}
        print(f"Wrote {video}", flush=True)
    write_json(OUT_DIR / "summary.json", summaries)
    zip_path = OUT_DIR / "control_response_lora_demo_v1_2026-06-28.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in OUT_DIR.rglob("*"):
            if path.is_file() and path != zip_path:
                z.write(path, path.relative_to(OUT_DIR))
    print(f"Wrote {zip_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--vad-threshold", type=float, default=0.01)
    parser.add_argument("--real-support-extra-s", type=float, default=2.0)
    parser.add_argument("--force-talker", action="store_true")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
