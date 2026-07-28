from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache/huggingface"
OUT_DIR = ROOT / "artifacts/omni_talker_vad_threshold_demo_v4"
CKPT_PATH = ROOT / "artifacts/omni3b_streaming_calibrated_tuning_v3/adapter_proxy_streaming_calibrated_v3_layer-20.pt"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
QWEN3TTS_PRIMARY_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN3TTS_FALLBACK_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
TICK_INTERVAL_S = 0.5
POST_SILENCE_S = 6.0
VAD_FRAME_S = 0.05
VAD_HOP_S = 0.02
VAD_RMS_THRESHOLD = 0.01

DEMO_SPECS = {
    "en_vulnerable": {
        "language_name": "English",
        "qwen3tts_language": "English",
        "demo_name": "English vulnerable strict audio-only delayed-backchannel demo",
        "pattern": "audio_only_partial_stream",
        "fragment": "Honestly, I want to tell you what happened after class, but I feel scared to say it",
        "profile": "vulnerable",
        "input_speaker": "Ryan",
        "input_instruct": "Speak gently and naturally, with a hesitant conversational tone. Do not add extra words.",
        "backchannel_instruction": "Give one very short natural acknowledgement, no more than four words.",
        "support_instruction": "Generate one short supportive sentence that fits the user's vulnerable fragment.",
    },
    "en_asked_wait": {
        "language_name": "English",
        "qwen3tts_language": "English",
        "demo_name": "English asked-wait strict audio-only delayed-backchannel demo",
        "pattern": "audio_only_partial_stream",
        "fragment": "Please give me a moment, I am still trying to put what happened after class into words",
        "profile": "asked_wait",
        "input_speaker": "Ryan",
        "input_instruct": "Speak calmly and naturally, as someone asking for a little time. Do not add extra words.",
        "backchannel_instruction": "Give one very short patient acknowledgement, no more than four words.",
        "support_instruction": "Generate one short patient sentence that respects the user's request for time.",
    },
}


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, hidden_size))

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, dim: int, hidden: int = 192, dropout: float = 0.12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(LABELS)),
        )

    def forward(self, x):
        return self.net(x)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.reshape(-1), int(sr)


def trim_audio(wav: np.ndarray, sr: int, threshold: float = 0.01, pad_s: float = 0.05) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0 or not np.any(np.abs(wav) > threshold):
        return wav
    idx = np.flatnonzero(np.abs(wav) > threshold)
    pad = int(sr * pad_s)
    lo = max(0, int(idx[0]) - pad)
    hi = min(len(wav), int(idx[-1]) + pad)
    return wav[lo:hi].astype(np.float32)


def fade_and_limit(wav: np.ndarray, sr: int, max_s: float | None = None) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if max_s is not None:
        limit = int(sr * max_s)
        if wav.size > limit:
            wav = wav[:limit].copy()
    fade = min(int(sr * 0.04), max(1, wav.size // 4))
    if wav.size > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.98:
        wav = wav / peak * 0.98
    return wav.astype(np.float32)


def load_qwen3tts(model_id: str = QWEN3TTS_PRIMARY_MODEL):
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation="eager",
    )
    return model


def ensure_input_audio(spec: dict, out_path: Path, tts_model=None, force: bool = False, tts_model_id: str = ""):
    if out_path.exists() and out_path.stat().st_size > 4096 and not force:
        meta_path = out_path.with_suffix(".json")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {"path": str(out_path), "reused": True, "model_id": tts_model_id}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    owns_model = tts_model is None
    if tts_model is None:
        tts_model = load_qwen3tts(QWEN3TTS_PRIMARY_MODEL)
        tts_model_id = QWEN3TTS_PRIMARY_MODEL
    wavs, sr = tts_model.generate_custom_voice(
        text=[spec["fragment"]],
        language=[spec["qwen3tts_language"]],
        speaker=[spec["input_speaker"]],
        instruct=[spec["input_instruct"]],
        max_new_tokens=360,
    )
    wav = trim_audio(np.asarray(wavs[0], dtype=np.float32).reshape(-1), sr, threshold=0.008)
    sf.write(str(out_path), fade_and_limit(wav, sr), sr)
    meta = {
        "path": str(out_path),
        "source": "Qwen3TTS generate_custom_voice",
        "model_id": tts_model_id,
        "speaker": spec["input_speaker"],
        "language": spec["qwen3tts_language"],
        "fragment_text_for_tts_only": spec["fragment"],
        "note": "This transcript is used only to synthesize input speech. It is not supplied to Omni hidden extraction or response generation.",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if owns_model:
        del tts_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return meta


def load_adapter_stack():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    adapter = FeatureAdapter(5, 2048)
    adapter.load_state_dict(ckpt["adapter_state"])
    adapter.eval()
    head = Head(4096)
    head.load_state_dict(ckpt["proxy_head_state"])
    head.eval()
    mean = np.asarray(ckpt["proxy_mean"], dtype=np.float32)
    std = np.asarray(ckpt["proxy_std"], dtype=np.float32)
    return ckpt, adapter, head, mean, std


def move_inputs(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        else:
            moved[key] = value
    return moved


def build_hidden_conversation(audio_path: Path):
    instruction = (
        "Listen to the audio provided so far in a streaming dialogue.\n"
        "No transcript or silence duration is provided in text.\n"
        "Choose exactly one listener timing label: WAIT, BACKCHANNEL, or SUPPORT. "
        "WAIT means keep listening. BACKCHANNEL means give a short acknowledgement. "
        "SUPPORT means actively respond or help. Output only the label."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM}]},
        {"role": "user", "content": [{"type": "audio", "audio": str(audio_path.resolve())}, {"type": "text", "text": instruction}]},
    ]


def extract_context_hidden(model, processor, dtype, input_audio: Path, layer: int) -> np.ndarray:
    conv = build_hidden_conversation(input_audio)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    moved = move_inputs(inputs, model.device, dtype)
    with torch.inference_mode():
        outputs = model.thinker(**moved, output_hidden_states=True, use_audio_in_video=False)
    hidden = outputs.hidden_states[layer + 1][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
    return hidden


def make_rows(spec: dict, speech_duration: float, total_duration: float) -> list[dict]:
    rows = []
    previous = None
    tick_count = int(math.floor(total_duration / TICK_INTERVAL_S)) + 1
    for idx in range(tick_count + 1):
        clock_s = round(idx * TICK_INTERVAL_S, 4)
        if clock_s > total_duration + 1e-6:
            continue
        delta_t = 0.0 if previous is None else round(clock_s - previous, 4)
        previous = clock_s
        is_speaking = clock_s < speech_duration - 1e-6
        silence_elapsed = 0.0 if is_speaking else round(max(0.0, clock_s - speech_duration), 4)
        utterance_elapsed = round(clock_s, 4)
        rows.append(
            {
                "clock_s": float(clock_s),
                "timepoint_s": float(silence_elapsed),
                "features": {
                    "silence_elapsed": float(silence_elapsed),
                    "delta_t": float(delta_t),
                    "utterance_elapsed": float(utterance_elapsed),
                    "is_user_speaking": bool(is_speaking),
                    "asr_changed": bool(is_speaking and clock_s > 0.0),
                },
                "profile": spec["profile"],
                "fragment": "[audio-only; transcript withheld from Omni]",
            }
        )
    return rows


def build_vad_triggered_rows(input_audio: Path, spec: dict) -> tuple[list[dict], np.ndarray, int, dict]:
    wav, sr = read_audio(input_audio)
    if sr != 24000:
        x_old = np.linspace(0, 1, len(wav), endpoint=False)
        x_new = np.linspace(0, 1, int(len(wav) * 24000 / sr), endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
        sr = 24000
    stream = np.concatenate([wav, np.zeros(int(round(POST_SILENCE_S * sr)), dtype=np.float32)]).astype(np.float32)
    frame = max(1, int(round(VAD_FRAME_S * sr)))
    hop = max(1, int(round(VAD_HOP_S * sr)))
    rows = []
    frames = []
    last_voice_time = 0.0
    last_tick_time = -999.0
    was_below = False
    utterance_start_time = None
    previous_tick_time = None
    for start in range(0, max(1, len(stream) - frame + 1), hop):
        end = min(len(stream), start + frame)
        frame_end_s = round(end / sr, 4)
        seg = stream[start:end]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        vad_speaking = bool(rms >= VAD_RMS_THRESHOLD)
        if vad_speaking:
            last_voice_time = frame_end_s
            if utterance_start_time is None:
                utterance_start_time = round(start / sr, 4)
            was_below = False
            frames.append({"time_s": frame_end_s, "rms": rms, "vad_speaking": True, "silence_elapsed": 0.0})
            continue
        silence_elapsed = round(max(0.0, frame_end_s - last_voice_time), 4)
        frames.append({"time_s": frame_end_s, "rms": rms, "vad_speaking": False, "silence_elapsed": silence_elapsed})
        should_tick = (not was_below) or (frame_end_s - last_tick_time >= TICK_INTERVAL_S - 1e-9)
        was_below = True
        if not should_tick:
            continue
        delta_t = 0.0 if previous_tick_time is None else round(frame_end_s - previous_tick_time, 4)
        previous_tick_time = frame_end_s
        last_tick_time = frame_end_s
        utterance_elapsed = frame_end_s if utterance_start_time is None else round(frame_end_s - utterance_start_time, 4)
        rows.append(
            {
                "clock_s": float(frame_end_s),
                "timepoint_s": float(silence_elapsed),
                "features": {
                    "silence_elapsed": float(silence_elapsed),
                    "delta_t": float(delta_t),
                    "utterance_elapsed": float(utterance_elapsed),
                    "is_user_speaking": False,
                    "asr_changed": bool(any(f["vad_speaking"] for f in frames if previous_tick_time is None or f["time_s"] >= max(0.0, frame_end_s - max(delta_t, VAD_HOP_S) - VAD_FRAME_S))),
                },
                "profile": spec["profile"],
                "fragment": "[audio-only; transcript withheld from Omni]",
                "vad_rms": rms,
                "vad_threshold": VAD_RMS_THRESHOLD,
                "vad_frame_s": VAD_FRAME_S,
                "vad_hop_s": VAD_HOP_S,
                "vad_trigger": "rms_below_threshold",
                "last_voice_time_s": float(last_voice_time),
            }
        )
    vad_meta = {
        "vad_type": "fixed_rms_threshold",
        "rms_threshold": VAD_RMS_THRESHOLD,
        "frame_s": VAD_FRAME_S,
        "hop_s": VAD_HOP_S,
        "input_speech_duration_s": float(len(wav) / sr),
        "stream_duration_s": float(len(stream) / sr),
        "post_silence_s": POST_SILENCE_S,
        "tick_interval_while_below_threshold_s": TICK_INTERVAL_S,
        "frames": frames,
        "note": "Ticks are created only from observed RMS crossing below the threshold; input speech duration is not used to compute silence_elapsed.",
    }
    return rows, stream, sr, vad_meta


def write_vad_stream_segments(stream: np.ndarray, sr: int, rows: list[dict], out_dir: Path) -> tuple[list[dict], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for row in rows:
        clock_s = float(row["clock_s"])
        end = max(1, min(len(stream), int(round(clock_s * sr))))
        segment = stream[:end].astype(np.float32)
        path = out_dir / f"vad_tick_{int(round(clock_s * 1000)):06d}ms.wav"
        sf.write(str(path), segment, sr)
        updated.append(
            {
                **row,
                "segment_audio_path": str(path),
                "segment_duration_s": float(len(segment) / sr),
            }
        )
    return updated, sr


def feature_matrix(rows: list[dict]) -> np.ndarray:
    vals = []
    for row in rows:
        f = row["features"]
        vals.append(
            [
                np.log1p(f["silence_elapsed"]),
                f["delta_t"],
                np.log1p(f["utterance_elapsed"]),
                1.0 if f["is_user_speaking"] else 0.0,
                1.0 if f["asr_changed"] else 0.0,
            ]
        )
    return np.asarray(vals, dtype=np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def write_stream_segments(input_audio: Path, rows: list[dict], speech_duration: float, out_dir: Path) -> tuple[list[dict], int]:
    wav, sr = read_audio(input_audio)
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for row in rows:
        clock_s = float(row["clock_s"])
        if clock_s <= 0.0:
            row = {**row, "segment_audio_path": "", "segment_duration_s": 0.0}
            updated.append(row)
            continue
        if clock_s < speech_duration:
            end = max(1, min(len(wav), int(round(clock_s * sr))))
            segment = wav[:end]
        else:
            silence = np.zeros(int(round((clock_s - speech_duration) * sr)), dtype=np.float32)
            segment = np.concatenate([wav, silence]).astype(np.float32)
        path = out_dir / f"tick_{int(round(clock_s * 1000)):06d}ms.wav"
        sf.write(str(path), segment, sr)
        row = {
            **row,
            "segment_audio_path": str(path),
            "segment_duration_s": float(len(segment) / sr),
        }
        updated.append(row)
    return updated, sr


def extract_hidden_sequence(model, processor, dtype, rows: list[dict], layer: int) -> np.ndarray:
    hiddens = []
    for row in rows:
        if not row["segment_audio_path"]:
            hiddens.append(np.zeros(2048, dtype=np.float32))
            continue
        hidden = extract_context_hidden(model, processor, dtype, Path(row["segment_audio_path"]), layer)
        hiddens.append(hidden)
    return np.stack(hiddens, axis=0).astype(np.float32)


def decide_sequence(spec: dict, context_hiddens: np.ndarray, rows: list[dict]):
    ckpt, adapter, head, mean, std = load_adapter_stack()
    with torch.no_grad():
        delta = adapter(torch.tensor(feature_matrix(rows), dtype=torch.float32)).cpu().numpy().astype(np.float32)
        context = context_hiddens.astype(np.float32)
        feats = np.concatenate([context, delta], axis=1).astype(np.float32)
        logits = head(torch.tensor(apply_standardizer(feats, mean, std), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    centered = delta - delta.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vh[:2].T
    except Exception:
        xy = np.zeros((len(rows), 2), dtype=np.float32)
    if xy.shape[1] < 2:
        xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
    delta_norms = np.linalg.norm(delta, axis=1)
    context_norms = np.linalg.norm(context, axis=1)
    cos = np.sum(delta * context, axis=1) / np.maximum(delta_norms * context_norms, 1e-12)
    step_l2 = np.zeros(len(rows), dtype=np.float32)
    if len(rows) > 1:
        step_l2[1:] = np.linalg.norm(delta[1:] - delta[:-1], axis=1)

    out = []
    for i, (row, prob) in enumerate(zip(rows, probs)):
        if float(row["clock_s"]) <= 0.0:
            prob = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            raw = "WAIT"
        else:
            raw = LABELS[int(np.argmax(prob))]
        emitted = raw
        reason = "startup_no_audio" if float(row["clock_s"]) <= 0.0 else "learned_argmax_no_runtime_gate"
        out.append(
            {
                **row,
                "raw_label": raw,
                "emitted_action": emitted,
                "policy_reason": reason,
                "p_WAIT": float(prob[0]),
                "p_BACKCHANNEL": float(prob[1]),
                "p_SUPPORT": float(prob[2]),
                "context_norm": float(context_norms[i]),
                "time_vector_norm": float(delta_norms[i]),
                "time_context_cosine": float(cos[i]),
                "time_vector_step_l2": float(step_l2[i]),
                "time_vector_pca_x": float(xy[i, 0]),
                "time_vector_pca_y": float(xy[i, 1]),
                "layer": int(ckpt["layer"]),
            }
        )
    return out


def build_talker_prompt(spec: dict, action: str) -> str:
    action_instruction = spec["backchannel_instruction"] if action == "BACKCHANNEL" else spec["support_instruction"]
    return (
        "You are the listener in a spoken English dialogue.\n"
        "The user's audio so far is attached. No transcript is provided.\n"
        f"The timing controller selected {action}.\n"
        f"{action_instruction}\n"
        "Return only the words to speak aloud in English. "
        "Do not mention WAIT, BACKCHANNEL, SUPPORT, labels, timers, versions, or system details. "
        "Do not use parentheses, brackets, markdown, stage directions, or quoted metadata."
    )


def invalid_spoken_text(text: str) -> bool:
    lowered = text.lower()
    if any(token in text for token in ["WAIT", "BACKCHANNEL", "SUPPORT"]):
        return True
    banned = [
        "label",
        "timer",
        "timing controller",
        "version",
        "v2",
        "v3",
        "assistant",
        "system",
    ]
    if any(token in lowered for token in banned):
        return True
    if any(ch in text for ch in "()[]{}<>"):
        return True
    return "\n" in text or len(text.strip()) == 0


def audio_metrics(wav: np.ndarray, sr: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return {"duration_s": 0.0, "rms_mean": 0.0, "rms_std": 0.0, "zcr_mean": 0.0, "zcr_std": 0.0, "peak": 0.0}
    frame = max(1, int(sr * 0.05))
    rms_vals, zcr_vals = [], []
    for start in range(0, max(1, len(wav) - frame), frame):
        seg = wav[start : start + frame]
        if seg.size == 0:
            continue
        rms_vals.append(float(np.sqrt(np.mean(seg * seg))))
        if seg.size > 1:
            zcr_vals.append(float(np.mean(np.abs(np.diff(np.signbit(seg))).astype(np.float32))))
    rms = np.asarray(rms_vals or [0.0], dtype=np.float32)
    zcr = np.asarray(zcr_vals or [0.0], dtype=np.float32)
    return {
        "duration_s": float(len(wav) / sr),
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
        "peak": float(np.max(np.abs(wav))),
    }


def suspicious_talker_audio(metrics: dict, action: str, limited: bool) -> bool:
    if limited:
        return True
    if metrics["duration_s"] > 1.0 and metrics["rms_std"] < 0.018 and metrics["zcr_std"] < 0.018:
        return True
    if metrics["duration_s"] > 2.0 and metrics["rms_mean"] > 0.18 and metrics["zcr_std"] < 0.035:
        return True
    return False


def generate_omni_talker_response(
    model,
    processor,
    spec: dict,
    action: str,
    out_path: Path,
    speaker: str,
    context_audio_path: Path,
    force: bool = False,
) -> dict:
    if out_path.exists() and out_path.stat().st_size > 4096 and not force:
        meta_path = out_path.with_suffix(".json")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
    prompt = build_talker_prompt(spec, action)
    conv = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM}]},
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
    inputs = move_inputs(inputs, model.device, model.dtype)
    max_text = 20 if action == "BACKCHANNEL" else 48
    max_audio_s = 2.0 if action == "BACKCHANNEL" else 5.0
    attempts = []
    selected = None
    for attempt in range(3):
        torch.manual_seed(20260625 + attempt)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260625 + attempt)
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
        trimmed = trim_audio(raw_wav, 24000, threshold=0.006)
        limited = len(trimmed) > int(max_audio_s * 24000)
        wav = fade_and_limit(trimmed, 24000, max_s=max_audio_s)
        metrics = audio_metrics(wav, 24000)
        rejected = invalid_spoken_text(assistant_text) or suspicious_talker_audio(metrics, action, limited)
        attempts.append(
            {
                "attempt": attempt,
                "assistant_text": assistant_text,
                "limited": limited,
                "metrics": metrics,
                "rejected": rejected,
            }
        )
        selected = (assistant_text, wav, metrics)
        if not rejected:
            break
    assistant_text, wav, metrics = selected
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
    out_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def mix_audio(input_audio: Path, rows: list[dict], response_meta: dict[str, dict], out_wav: Path) -> dict:
    user, sr = read_audio(input_audio)
    if sr != 24000:
        x_old = np.linspace(0, 1, len(user), endpoint=False)
        x_new = np.linspace(0, 1, int(len(user) * 24000 / sr), endpoint=False)
        user = np.interp(x_new, x_old, user).astype(np.float32)
        sr = 24000
    speech_duration = len(user) / sr
    final_duration = max(row["clock_s"] for row in rows) + 4.0
    track = np.zeros(int(math.ceil(final_duration * sr)), dtype=np.float32)
    track[: len(user)] += user * 0.92
    events = []
    last_spoken_action = "WAIT"
    for row in rows:
        action = row["emitted_action"]
        row["response_text"] = ""
        row["playback_status"] = "no_speech_for_wait"
        if action == "WAIT":
            last_spoken_action = "WAIT"
            continue
        if action not in response_meta:
            row["playback_status"] = "no_response_audio_available"
            continue
        if action == last_spoken_action:
            row["response_text"] = response_meta[action]["assistant_text"]
            row["playback_status"] = "same_label_not_replayed_in_audio_mix"
            continue
        last_spoken_action = action
        wav, wav_sr = read_audio(Path(response_meta[action]["path"]))
        if wav_sr != sr:
            x_old = np.linspace(0, 1, len(wav), endpoint=False)
            x_new = np.linspace(0, 1, int(len(wav) * sr / wav_sr), endpoint=False)
            wav = np.interp(x_new, x_old, wav).astype(np.float32)
        start_s = float(row["clock_s"])
        start = int(round(start_s * sr))
        end = min(len(track), start + len(wav))
        if end > start:
            track[start:end] += wav[: end - start] * 0.9
        events.append(
            {
                "action": action,
                "start_s": start_s,
                "duration_s": (end - start) / sr,
                "assistant_text": response_meta[action]["assistant_text"],
                "speaker": response_meta[action]["speaker"],
            }
        )
        row["response_text"] = response_meta[action]["assistant_text"]
        row["playback_status"] = "spoken_on_label_transition"
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track = track / peak * 0.98
    sf.write(str(out_wav), track, sr)
    return {"sample_rate": sr, "speech_duration_s": speech_duration, "duration_s": len(track) / sr, "events": events}


def get_font(size: int):
    for path in ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/arial.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if len(words) <= 1:
        chars = list(text)
        words = chars
        sep = ""
    else:
        sep = " "
    lines, current = [], ""
    for word in words:
        trial = word if not current else current + sep + word
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_bar(draw, x, y, width, label, prob, color, font):
    draw.text((x, y), label, fill=(230, 233, 238), font=font)
    bx, by = x + 170, y + 4
    draw.rectangle((bx, by, bx + width, by + 20), fill=(48, 54, 65), outline=(100, 108, 122))
    draw.rectangle((bx, by, bx + int(width * prob), by + 20), fill=color)
    draw.text((bx + width + 12, y), f"{prob:.2f}", fill=(230, 233, 238), font=font)


def render_video(spec: dict, input_audio: Path, rows: list[dict], audio_meta: dict, response_meta: dict, out_mp4: Path):
    fps, w, h = 10, 1280, 720
    noaudio_path = out_mp4.with_name("demo_noaudio.mp4")
    duration = float(audio_meta["duration_s"])
    speech_duration = float(audio_meta["speech_duration_s"])
    events = audio_meta["events"]
    vad_frames = audio_meta.get("vad", {}).get("frames", [])
    vad_threshold = float(audio_meta.get("vad", {}).get("rms_threshold", VAD_RMS_THRESHOLD))
    title_font, font, small = get_font(30), get_font(24), get_font(18)
    rows_sorted = sorted(rows, key=lambda r: r["clock_s"])
    xs = np.asarray([r["time_vector_pca_x"] for r in rows_sorted], dtype=np.float32)
    ys = np.asarray([r["time_vector_pca_y"] for r in rows_sorted], dtype=np.float32)
    max_x, max_y = max(float(np.max(np.abs(xs))), 1e-6), max(float(np.max(np.abs(ys))), 1e-6)
    colors = {"WAIT": (102, 194, 165), "BACKCHANNEL": (252, 141, 98), "SUPPORT": (141, 160, 203)}

    writer = imageio.get_writer(str(noaudio_path), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    for frame_idx in range(int(math.ceil(duration * fps))):
        now = frame_idx / fps
        row = rows_sorted[0]
        for candidate in rows_sorted:
            if candidate["clock_s"] <= now + 1e-6:
                row = candidate
        vad_frame = None
        for candidate in vad_frames:
            if float(candidate["time_s"]) <= now + 1e-6:
                vad_frame = candidate
            else:
                break
        silence_elapsed = float(row["features"]["silence_elapsed"])
        current_event = next((e for e in events if e["start_s"] <= now <= e["start_s"] + e["duration_s"]), None)
        img = Image.new("RGB", (w, h), (18, 22, 30))
        draw = ImageDraw.Draw(img)
        draw.text((48, 34), "VAD-Threshold Realtime Demo v4", fill=(248, 249, 251), font=title_font)
        draw.text(
            (50, 86),
            f"input voice: Qwen3TTS {spec['input_speaker']} / output voice: Qwen2.5-Omni Talker",
            fill=(178, 188, 204),
            font=small,
        )
        draw.text((50, 132), "User audio content", fill=(141, 211, 199), font=font)
        y = 168
        display_fragment = "Transcript is withheld from Omni. Text shown here is for the human viewer only: " + spec["fragment"]
        for line in wrap_text(draw, display_fragment, small, 760)[:4]:
            draw.text((50, y), line, fill=(235, 238, 244), font=small)
            y += 25
        draw.text((860, 132), f"clock: {now:05.2f}s", fill=(248, 249, 251), font=font)
        draw.text((860, 168), f"silence_elapsed: {silence_elapsed:04.2f}s", fill=(248, 249, 251), font=font)
        if current_event:
            vad = "OMNI TALKER SPEAKING"
        elif vad_frame is None:
            vad = "VAD WARMUP"
        else:
            vad = "VAD SPEECH" if vad_frame.get("vad_speaking") else "VAD BELOW THRESHOLD"
        draw.text((860, 204), f"VAD: {vad}", fill=(255, 213, 128), font=small)
        draw.text((860, 232), f"speaker: {current_event['speaker'] if current_event else '-'}", fill=(255, 213, 128), font=small)
        rms_now = 0.0 if vad_frame is None else float(vad_frame.get("rms", 0.0))
        draw.text((860, 258), f"RMS: {rms_now:.4f} / threshold {vad_threshold:.4f}", fill=(255, 213, 128), font=small)

        y = 285
        draw.text((50, y), "Decision head probabilities", fill=(141, 211, 199), font=font)
        y += 42
        draw_bar(draw, 50, y, 360, "WAIT", float(row["p_WAIT"]), colors["WAIT"], small)
        y += 42
        draw_bar(draw, 50, y, 360, "BACKCHANNEL", float(row["p_BACKCHANNEL"]), colors["BACKCHANNEL"], small)
        y += 42
        draw_bar(draw, 50, y, 360, "SUPPORT", float(row["p_SUPPORT"]), colors["SUPPORT"], small)
        y += 66
        draw.text((50, y), f"raw label: {row['raw_label']}", fill=(248, 249, 251), font=font)
        y += 34
        draw.text((50, y), f"emitted action: {row['emitted_action']}  ({row['policy_reason']})", fill=(255, 242, 174), font=font)
        y += 36
        draw.text((50, y), f"playback: {row.get('playback_status', '-')}", fill=(178, 188, 204), font=small)
        y += 26
        resp = current_event["assistant_text"] if current_event else row.get("response_text", "")
        for line in wrap_text(draw, f"Omni Talker text: {resp or '-'}", small, 760)[:2]:
            draw.text((50, y), line, fill=(235, 238, 244), font=small)
            y += 24

        panel = (850, 286, 1230, 560)
        draw.rounded_rectangle(panel, radius=8, fill=(27, 32, 43), outline=(83, 92, 110), width=1)
        draw.text((870, 302), "Internal movement: Time Adapter vector", fill=(141, 211, 199), font=small)
        plot = (890, 346, 1135, 525)
        draw.rectangle(plot, fill=(18, 22, 30), outline=(75, 84, 100))
        cx, cy = (plot[0] + plot[2]) / 2, (plot[1] + plot[3]) / 2
        draw.line((plot[0], cy, plot[2], cy), fill=(55, 62, 76), width=1)
        draw.line((cx, plot[1], cx, plot[3]), fill=(55, 62, 76), width=1)

        def point_for(r):
            return (
                int(cx + (float(r["time_vector_pca_x"]) / max_x) * ((plot[2] - plot[0]) * 0.43)),
                int(cy - (float(r["time_vector_pca_y"]) / max_y) * ((plot[3] - plot[1]) * 0.43)),
            )

        points = [point_for(r) for r in rows_sorted]
        for idx, (a, b) in enumerate(zip(points, points[1:])):
            line_color = (115, 125, 145) if rows_sorted[idx + 1]["clock_s"] <= now + 1e-6 else (55, 62, 76)
            draw.line((a[0], a[1], b[0], b[1]), fill=line_color, width=2)
        for r, p in zip(rows_sorted, points):
            col = colors.get(r["emitted_action"], (220, 220, 220)) if r["clock_s"] <= now + 1e-6 else (65, 72, 86)
            radius = 5 if r is not row else 9
            draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=col, outline=(248, 249, 251) if r is row else None)
        draw.text((870, 532), f"time_vec_norm={row['time_vector_norm']:.2f}  hidden_norm={row['context_norm']:.2f}", fill=(230, 233, 238), font=small)
        draw.text((870, 556), f"cos(time, hidden)={row['time_context_cosine']:.3f}  step_L2={row['time_vector_step_l2']:.2f}", fill=(230, 233, 238), font=small)

        rx0, ry0, rx1, ry1 = 50, 582, 1230, 610
        draw.rectangle((rx0, ry0, rx1, ry1), fill=(27, 32, 43), outline=(83, 92, 110))
        draw.text((50, 560), "VAD RMS threshold trigger", fill=(141, 211, 199), font=small)
        max_rms = max([float(f.get("rms", 0.0)) for f in vad_frames] + [vad_threshold * 2, 1e-6])
        thresh_y = ry1 - int(min(1.0, vad_threshold / max_rms) * (ry1 - ry0))
        draw.line((rx0, thresh_y, rx1, thresh_y), fill=(255, 110, 110), width=2)
        prev_pt = None
        for vf in vad_frames:
            t_v = float(vf["time_s"])
            if t_v > now + 1e-6:
                break
            x = rx0 + int((t_v / duration) * (rx1 - rx0))
            y_v = ry1 - int(min(1.0, float(vf.get("rms", 0.0)) / max_rms) * (ry1 - ry0))
            pt = (x, y_v)
            if prev_pt is not None:
                draw.line((prev_pt[0], prev_pt[1], pt[0], pt[1]), fill=(255, 213, 128), width=2)
            prev_pt = pt
        draw.text((1030, 560), f"threshold={vad_threshold:.3f}", fill=(255, 150, 150), font=small)

        tx0, ty0, tx1, ty1 = 50, 620, 1230, 648
        draw.rectangle((tx0, ty0, tx1, ty1), fill=(45, 51, 62), outline=(110, 119, 134))
        speech_x = tx0 + int((speech_duration / duration) * (tx1 - tx0))
        draw.rectangle((tx0, ty0, speech_x, ty1), fill=(65, 145, 170))
        for r in rows_sorted:
            x = tx0 + int((r["clock_s"] / duration) * (tx1 - tx0))
            col = colors.get(r["emitted_action"], (248, 249, 251)) if r["clock_s"] <= now + 1e-6 else (105, 113, 130)
            draw.line((x, ty0 - 12, x, ty1 + 12), fill=col, width=2)
        now_x = tx0 + int((now / duration) * (tx1 - tx0))
        draw.line((now_x, ty0 - 28, now_x, ty1 + 28), fill=(255, 255, 255), width=3)
        draw.text((50, 656), "blue=input voice, green/orange/purple=VAD-triggered decisions; ticks occur only while RMS is below threshold", fill=(178, 188, 204), font=small)
        writer.append_data(np.asarray(img))
    writer.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(noaudio_path),
            "-i",
            str(out_mp4.with_suffix(".wav")),
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_csv(path: Path, rows: list[dict]):
    fields = [
        "clock_s",
        "timepoint_s",
        "segment_audio_path",
        "segment_duration_s",
        "is_user_speaking",
        "silence_elapsed",
        "utterance_elapsed",
        "vad_rms",
        "vad_threshold",
        "vad_trigger",
        "last_voice_time_s",
        "raw_label",
        "emitted_action",
        "policy_reason",
        "playback_status",
        "p_WAIT",
        "p_BACKCHANNEL",
        "p_SUPPORT",
        "response_text",
        "time_vector_norm",
        "context_norm",
        "time_context_cosine",
        "time_vector_step_l2",
        "time_vector_pca_x",
        "time_vector_pca_y",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                **row,
                "is_user_speaking": row.get("features", {}).get("is_user_speaking", ""),
                "silence_elapsed": row.get("features", {}).get("silence_elapsed", ""),
                "utterance_elapsed": row.get("features", {}).get("utterance_elapsed", ""),
            }
            writer.writerow({k: flat.get(k, "") for k in fields})


def load_omni():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    return model, processor, dtype


def run_demo(
    demo_key: str,
    speaker: str,
    model,
    processor,
    dtype,
    tts_model=None,
    force_input: bool = False,
    force_talker: bool = False,
):
    spec = DEMO_SPECS[demo_key]
    out_dir = OUT_DIR / demo_key
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / f"input_qwen3tts_{spec['input_speaker'].lower()}.wav"
    input_meta = ensure_input_audio(spec, input_path, tts_model=tts_model, force=force_input)

    ckpt, _, _, _, _ = load_adapter_stack()
    rows, stream, stream_sr, vad_meta = build_vad_triggered_rows(input_path, spec)
    rows, segment_sr = write_vad_stream_segments(stream, stream_sr, rows, out_dir / "segments")
    hiddens = extract_hidden_sequence(model, processor, dtype, rows, int(ckpt["layer"]))
    rows = decide_sequence(spec, hiddens, rows)

    response_meta = {}
    for action in ["BACKCHANNEL", "SUPPORT"]:
        selected_row = next((row for row in rows if row["emitted_action"] == action and row["segment_audio_path"]), None)
        if selected_row is not None:
            response_meta[action] = generate_omni_talker_response(
                model,
                processor,
                spec,
                action,
                out_dir / f"omni_talker_{speaker.lower()}_{action.lower()}.wav",
                speaker=speaker,
                context_audio_path=Path(selected_row["segment_audio_path"]),
                force=force_talker,
            )

    wav_path = out_dir / "demo.wav"
    audio_meta = mix_audio(input_path, rows, response_meta, wav_path)
    audio_meta["vad"] = vad_meta
    mp4_path = out_dir / "demo.mp4"
    render_video(spec, input_path, rows, audio_meta, response_meta, mp4_path)
    write_csv(out_dir / "decision_log.csv", rows)
    summary = {
        "demo_key": demo_key,
        "language": spec["language_name"],
        "pattern": spec["pattern"],
        "spec": spec,
        "input_audio": str(input_path),
        "input_audio_source": "Qwen3TTS generate_custom_voice",
        "input_audio_tts": input_meta,
        "omni_input_mode": "VAD-threshold-triggered strict audio-only context plus generic classification instruction; no transcript text is supplied to Omni",
        "tick_interval_s": TICK_INTERVAL_S,
        "post_silence_s": POST_SILENCE_S,
        "vad_policy": {
            "type": "fixed_rms_threshold",
            "rms_threshold": VAD_RMS_THRESHOLD,
            "frame_s": VAD_FRAME_S,
            "hop_s": VAD_HOP_S,
            "tick_rule": "create a decision tick when frame RMS falls below threshold, then every 0.5s while it remains below",
            "timer_rule": "silence_elapsed is measured from the last frame whose RMS was at or above threshold",
        },
        "decision_policy": "learned decision head argmax only; VAD controls tick/timer, but no support-while-speaking mask, no backchannel cooldown, no confidence threshold, no support one-shot gate",
        "audio_playback_policy": "for readability, the mixed demo audio plays non-WAIT speech only on label transitions; this does not alter emitted_action in the decision log",
        "segment_sample_rate": segment_sr,
        "omni_talker_speaker": speaker,
        "checkpoint": str(CKPT_PATH),
        "audio": audio_meta,
        "response_meta": response_meta,
        "outputs": {
            "mp4": str(mp4_path),
            "wav": str(wav_path),
            "decision_log": str(out_dir / "decision_log.csv"),
        },
        "rows": rows,
        "notes": [
            "Input voice is generated by Qwen3TTS generate_custom_voice; no SAPI fallback is used.",
            "The default input TTS is the training-matched Qwen3TTS 0.6B CustomVoice checkpoint; the earlier 1.7B trial is preserved separately because it shifted the audio-hidden distribution.",
            "Omni receives audio segments only, plus a generic task instruction. The transcript is not supplied to Omni hidden extraction or response generation.",
            "Each decision tick is triggered by observed VAD threshold crossing, then uses the audio stream available up to that tick; hidden states are not reused from the full utterance.",
            "The speech end time is not used to compute silence_elapsed. The timer is derived from last_voice_time_s under the fixed RMS VAD threshold.",
            "BACKCHANNEL and SUPPORT spoken responses are generated by Qwen2.5-Omni-3B Talker from the selected tick audio segment with sampled talker decoding.",
            "The decision log preserves every raw learned action. No runtime timing gate changes BACKCHANNEL or SUPPORT into WAIT.",
            "The mixed audio avoids replaying the same non-WAIT label on consecutive ticks, only to keep the MP4 intelligible.",
            "The video shows transcript text only for the human viewer.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_input_audios(demo_keys: list[str], force_input: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attempts = []
    for model_id in [QWEN3TTS_PRIMARY_MODEL, QWEN3TTS_FALLBACK_MODEL]:
        tts_model = None
        try:
            tts_model = load_qwen3tts(model_id)
            metas = {}
            for demo_key in demo_keys:
                spec = DEMO_SPECS[demo_key]
                out_dir = OUT_DIR / demo_key
                out_dir.mkdir(parents=True, exist_ok=True)
                input_path = out_dir / f"input_qwen3tts_{spec['input_speaker'].lower()}.wav"
                metas[demo_key] = ensure_input_audio(
                    spec,
                    input_path,
                    tts_model=tts_model,
                    force=force_input,
                    tts_model_id=model_id,
                )
            result = {
                "selected_model_id": model_id,
                "fallback_used": model_id != QWEN3TTS_PRIMARY_MODEL,
                "attempts": attempts,
                "inputs": metas,
            }
            (OUT_DIR / "qwen3tts_input_generation_log.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            attempts.append({"model_id": model_id, "error": repr(exc)})
            if tts_model is not None:
                del tts_model
                tts_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if model_id == QWEN3TTS_FALLBACK_MODEL:
                failure = {"selected_model_id": None, "fallback_used": True, "attempts": attempts}
                (OUT_DIR / "qwen3tts_input_generation_log.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
        finally:
            if tts_model is not None:
                del tts_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("Qwen3TTS input generation failed for all configured Qwen3TTS models.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=[*DEMO_SPECS.keys(), "all"], default="all")
    parser.add_argument("--speaker", default="Chelsie", choices=["Ethan", "Chelsie"])
    parser.add_argument("--force-input", action="store_true")
    parser.add_argument("--force-talker", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    outputs = {}
    demo_keys = list(DEMO_SPECS.keys()) if args.demo == "all" else [args.demo]
    input_generation = prepare_input_audios(demo_keys, force_input=args.force_input)

    model, processor, dtype = load_omni()
    try:
        for demo_key in demo_keys:
            summary = run_demo(
                demo_key,
                args.speaker,
                model,
                processor,
                dtype,
                tts_model=None,
                force_input=False,
                force_talker=args.force_talker,
            )
            outputs[demo_key] = summary["outputs"]
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"input_generation": input_generation, "outputs": outputs, "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
