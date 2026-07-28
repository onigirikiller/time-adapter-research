from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/omni3b_sequential_v2"
HIDDEN_PATH = ROOT / "artifacts/omni3b_sequential_v2/hidden_cache/no_time_test.npy"
CKPT_PATH = ROOT / "artifacts/omni3b_generation_hook_v3/adapter_proxy_stage-extra_layer-8.pt"
OUT_DIR = ROOT / "artifacts/realtime_decision_demo_v1"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows():
    rows = read_jsonl(DATA_DIR / "test.jsonl")
    by_context: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_context[row["context_id"]].append((idx, row))
    for context_id in list(by_context):
        by_context[context_id] = sorted(by_context[context_id], key=lambda x: x[1]["silence_seconds"])
    return rows, by_context


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


def load_stack():
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


def predict_context(context_id: str, hidden_mode: str = "frozen_0s") -> tuple[dict, list[dict]]:
    _, by_context = load_rows()
    if context_id not in by_context:
        raise KeyError(f"Unknown context_id: {context_id}")
    ckpt, adapter, head, mean, std = load_stack()
    hidden = np.load(HIDDEN_PATH, mmap_mode="r")
    pairs = by_context[context_id]
    seq_rows = [row for _, row in pairs]

    with torch.no_grad():
        delta = adapter(torch.tensor(feature_matrix(seq_rows), dtype=torch.float32)).cpu().numpy().astype(np.float32)

    if hidden_mode == "frozen_0s":
        base_index = pairs[0][0]
        context_hidden = np.repeat(np.asarray(hidden[base_index, ckpt["layer"], :], dtype=np.float32)[None, :], len(seq_rows), axis=0)
    elif hidden_mode == "recomputed_each_timepoint":
        context_hidden = np.stack([np.asarray(hidden[i, ckpt["layer"], :], dtype=np.float32) for i, _ in pairs], axis=0)
    else:
        raise ValueError(f"Unsupported hidden_mode: {hidden_mode}")

    decision_features = np.concatenate([context_hidden, delta], axis=1).astype(np.float32)
    with torch.no_grad():
        logits = head(torch.tensor(apply_standardizer(decision_features, mean, std), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    delta_centered = delta - delta.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(delta_centered, full_matrices=False)
        delta_xy = delta_centered @ vh[:2].T
    except Exception:
        delta_xy = np.zeros((len(seq_rows), 2), dtype=np.float32)
    if delta_xy.shape[1] < 2:
        delta_xy = np.pad(delta_xy, ((0, 0), (0, 2 - delta_xy.shape[1])))
    delta_norms = np.linalg.norm(delta, axis=1)
    context_norms = np.linalg.norm(context_hidden, axis=1)
    denom = np.maximum(delta_norms * context_norms, 1e-12)
    delta_context_cos = np.sum(delta * context_hidden, axis=1) / denom
    delta_step_l2 = np.zeros(len(delta), dtype=np.float32)
    if len(delta) > 1:
        delta_step_l2[1:] = np.linalg.norm(delta[1:] - delta[:-1], axis=1)

    out_rows = []
    last_backchannel_t = -999.0
    support_emitted = False
    for row, prob in zip(seq_rows, probs):
        raw = LABELS[int(np.argmax(prob))]
        emitted = raw
        reason = "raw"
        silence = float(row["silence_seconds"])
        is_user_speaking = bool(row["features"]["is_user_speaking"])

        if is_user_speaking and raw == "SUPPORT":
            emitted = "WAIT"
            reason = "masked_support_while_user_speaking"
        if raw == "BACKCHANNEL" and silence - last_backchannel_t < 2.0:
            emitted = "WAIT"
            reason = "backchannel_cooldown"
        if raw == "BACKCHANNEL" and prob[1] < 0.55:
            emitted = "WAIT"
            reason = "backchannel_low_confidence"
        if raw == "SUPPORT" and support_emitted:
            emitted = "WAIT"
            reason = "support_one_shot"

        response_text = ""
        if emitted == "BACKCHANNEL":
            response_text = "Mm-hm."
            last_backchannel_t = silence
        elif emitted == "SUPPORT":
            response_text = "Take your time. I'm listening."
            support_emitted = True

        out_rows.append(
            {
                "context_id": context_id,
                "profile": row["profile"],
                "fragment": row["fragment"],
                "timepoint_s": silence,
                "gold": row["label"],
                "raw_label": raw,
                "emitted_action": emitted,
                "policy_reason": reason,
                "p_WAIT": float(prob[0]),
                "p_BACKCHANNEL": float(prob[1]),
                "p_SUPPORT": float(prob[2]),
                "response_text": response_text,
                "hidden_mode": hidden_mode,
                "layer": int(ckpt["layer"]),
                "context_norm": float(context_norms[len(out_rows)]),
                "time_vector_norm": float(delta_norms[len(out_rows)]),
                "time_context_cosine": float(delta_context_cos[len(out_rows)]),
                "time_vector_step_l2": float(delta_step_l2[len(out_rows)]),
                "time_vector_pca_x": float(delta_xy[len(out_rows), 0]),
                "time_vector_pca_y": float(delta_xy[len(out_rows), 1]),
                "features": row["features"],
            }
        )
    meta = {
        "context_id": context_id,
        "profile": seq_rows[0]["profile"],
        "fragment": seq_rows[0]["fragment"],
        "base_audio_path": str(Path(seq_rows[0]["audio_path"])),
        "hidden_mode": hidden_mode,
        "checkpoint": str(CKPT_PATH),
        "layer": int(ckpt["layer"]),
        "labels": LABELS,
    }
    return meta, out_rows


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.reshape(-1), int(sr)


def trim_audio(audio: np.ndarray, threshold: float = 0.01, pad: int = 1600) -> np.ndarray:
    if not np.any(np.abs(audio) > threshold):
        return audio
    idx = np.flatnonzero(np.abs(audio) > threshold)
    lo = max(0, int(idx[0]) - pad)
    hi = min(len(audio), int(idx[-1]) + pad)
    return audio[lo:hi]


def synthesize_responses(response_dir: Path, force: bool = False) -> dict[str, Path]:
    response_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "BACKCHANNEL": response_dir / "backchannel_mmhm.wav",
        "SUPPORT": response_dir / "support_take_your_time.wav",
    }
    if all(path.exists() and path.stat().st_size > 4096 for path in paths.values()) and not force:
        return paths

    try:
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=dtype,
            attn_implementation="eager",
        )
        texts = ["Mm-hm.", "Take your time. I'm listening."]
        instruct = [
            "Speak as a very short, gentle listener backchannel.",
            "Speak as a short, warm support response without taking over.",
        ]
        wavs, sr = model.generate_custom_voice(
            text=texts,
            language=["English", "English"],
            speaker=["Ryan", "Ryan"],
            instruct=instruct,
            max_new_tokens=160,
        )
        for label, wav in zip(["BACKCHANNEL", "SUPPORT"], wavs):
            sf.write(str(paths[label]), trim_audio(np.asarray(wav, dtype=np.float32).reshape(-1)), sr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return paths
    except Exception as exc:
        # Fallback: write gentle tones so the video and interruption plumbing can still be inspected.
        sr = 24000
        for label, duration, hz in [("BACKCHANNEL", 0.32, 440.0), ("SUPPORT", 1.1, 330.0)]:
            t = np.arange(int(sr * duration), dtype=np.float32) / sr
            env = np.minimum(1.0, np.arange(len(t)) / max(1, int(sr * 0.04)))
            env *= np.minimum(1.0, np.arange(len(t))[::-1] / max(1, int(sr * 0.08)))
            wav = (0.12 * np.sin(2 * np.pi * hz * t) * env).astype(np.float32)
            sf.write(str(paths[label]), wav, sr)
        (response_dir / "tts_fallback_reason.txt").write_text(repr(exc), encoding="utf-8")
        return paths


def mix_audio(meta: dict, rows: list[dict], out_wav: Path, response_paths: dict[str, Path], interrupt_support_after: float | None = None) -> dict:
    user_path = Path(meta["base_audio_path"])
    # The row path contains zero silence for the first timepoint in this demo.
    user_audio, sr = read_audio(user_path)
    speech_duration = len(user_audio) / sr
    final_duration = speech_duration + max(row["timepoint_s"] for row in rows) + 2.8
    track = np.zeros(int(math.ceil(final_duration * sr)), dtype=np.float32)
    track[: len(user_audio)] += user_audio * 0.92

    events = []
    for row in rows:
        action = row["emitted_action"]
        if action not in {"BACKCHANNEL", "SUPPORT"}:
            continue
        resp, resp_sr = read_audio(response_paths[action])
        if resp_sr != sr:
            # The Qwen3TTS dataset and response model normally use 24 kHz. Keep fallback simple if not.
            x_old = np.linspace(0, 1, len(resp), endpoint=False)
            x_new = np.linspace(0, 1, int(len(resp) * sr / resp_sr), endpoint=False)
            resp = np.interp(x_new, x_old, resp).astype(np.float32)
        start_s = speech_duration + row["timepoint_s"]
        interrupted = False
        if action == "SUPPORT" and interrupt_support_after is not None:
            cut = max(1, int(interrupt_support_after * sr))
            if cut < len(resp):
                resp = resp[:cut]
                interrupted = True
        start = int(round(start_s * sr))
        end = min(len(track), start + len(resp))
        if end > start:
            track[start:end] += resp[: end - start] * 0.88
        events.append(
            {
                "action": action,
                "start_s": start_s,
                "duration_s": len(resp) / sr,
                "interrupted": interrupted,
                "response_text": row["response_text"],
            }
        )

    peak = float(np.max(np.abs(track))) if len(track) else 0.0
    if peak > 0.98:
        track = track / peak * 0.98
    sf.write(str(out_wav), track, sr)
    return {"sample_rate": sr, "speech_duration_s": speech_duration, "duration_s": len(track) / sr, "events": events}


def get_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str, prob: float, color: tuple[int, int, int], font):
    draw.text((x, y), label, fill=(230, 233, 238), font=font)
    bx, by = x + 170, y + 4
    draw.rectangle((bx, by, bx + width, by + 20), fill=(48, 54, 65), outline=(100, 108, 122))
    draw.rectangle((bx, by, bx + int(width * prob), by + 20), fill=color)
    draw.text((bx + width + 12, y), f"{prob:.2f}", fill=(230, 233, 238), font=font)


def render_video(meta: dict, rows: list[dict], audio_meta: dict, out_mp4: Path, out_video_noaudio: Path):
    fps = 10
    w, h = 1280, 720
    duration = float(audio_meta["duration_s"])
    speech_duration = float(audio_meta["speech_duration_s"])
    events = audio_meta["events"]

    font_title = get_font(34)
    font = get_font(24)
    font_small = get_font(18)
    font_mono = get_font(20)
    rows_sorted = sorted(rows, key=lambda r: r["timepoint_s"])
    pca_x = np.asarray([r["time_vector_pca_x"] for r in rows_sorted], dtype=np.float32)
    pca_y = np.asarray([r["time_vector_pca_y"] for r in rows_sorted], dtype=np.float32)
    max_abs_x = max(float(np.max(np.abs(pca_x))), 1e-6)
    max_abs_y = max(float(np.max(np.abs(pca_y))), 1e-6)
    action_colors = {
        "WAIT": (102, 194, 165),
        "BACKCHANNEL": (252, 141, 98),
        "SUPPORT": (141, 160, 203),
    }

    writer = imageio.get_writer(str(out_video_noaudio), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    frame_count = int(math.ceil(duration * fps))
    for frame_idx in range(frame_count):
        now = frame_idx / fps
        silence_elapsed = max(0.0, now - speech_duration)
        row = rows_sorted[0]
        for candidate in rows_sorted:
            if candidate["timepoint_s"] <= silence_elapsed + 1e-6:
                row = candidate
        current_event = next((e for e in events if e["start_s"] <= now <= e["start_s"] + e["duration_s"]), None)

        img = Image.new("RGB", (w, h), (18, 22, 30))
        draw = ImageDraw.Draw(img)
        draw.text((48, 34), "Realtime Time Adapter Decision Demo v1", fill=(248, 249, 251), font=font_title)
        draw.text((50, 86), f"context: {meta['context_id']} / profile: {meta['profile']} / hidden: {meta['hidden_mode']}", fill=(178, 188, 204), font=font_small)

        y = 132
        draw.text((50, y), "User fragment", fill=(141, 211, 199), font=font)
        y += 36
        for line in wrap_text(draw, meta["fragment"], font_small, 760)[:4]:
            draw.text((50, y), line, fill=(235, 238, 244), font=font_small)
            y += 25

        status = "USER SPEAKING" if now < speech_duration else "SILENCE / TIMER LOOP"
        vad = "AI SPEAKING" if current_event else ("USER SPEAKING" if now < speech_duration else "NO VOICE")
        draw.text((860, 132), f"clock: {now:05.2f}s", fill=(248, 249, 251), font=font)
        draw.text((860, 168), f"silence_elapsed: {silence_elapsed:04.2f}s", fill=(248, 249, 251), font=font)
        draw.text((860, 204), f"state: {status}", fill=(255, 213, 128), font=font_small)
        draw.text((860, 232), f"VAD: {vad}", fill=(255, 213, 128), font=font_small)

        y = 285
        draw.text((50, y), "Decision head probabilities", fill=(141, 211, 199), font=font)
        y += 42
        draw_bar(draw, 50, y, 360, "WAIT", float(row["p_WAIT"]), (102, 194, 165), font_small)
        y += 42
        draw_bar(draw, 50, y, 360, "BACKCHANNEL", float(row["p_BACKCHANNEL"]), (252, 141, 98), font_small)
        y += 42
        draw_bar(draw, 50, y, 360, "SUPPORT", float(row["p_SUPPORT"]), (141, 160, 203), font_small)

        y += 66
        draw.text((50, y), f"raw label: {row['raw_label']}", fill=(248, 249, 251), font=font)
        y += 34
        draw.text((50, y), f"emitted action: {row['emitted_action']}  ({row['policy_reason']})", fill=(255, 242, 174), font=font)
        y += 36
        response = current_event["response_text"] if current_event else row["response_text"]
        draw.text((50, y), f"response: {response or '-'}", fill=(235, 238, 244), font=font_small)

        # Internal movement panel: the Time Adapter vector trajectory compressed to 2D.
        panel = (850, 286, 1230, 560)
        draw.rounded_rectangle(panel, radius=8, fill=(27, 32, 43), outline=(83, 92, 110), width=1)
        draw.text((870, 302), "Internal movement: Time Adapter vector", fill=(141, 211, 199), font=font_small)
        plot = (890, 346, 1135, 525)
        draw.rectangle(plot, fill=(18, 22, 30), outline=(75, 84, 100))
        cx = (plot[0] + plot[2]) / 2
        cy = (plot[1] + plot[3]) / 2
        draw.line((plot[0], cy, plot[2], cy), fill=(55, 62, 76), width=1)
        draw.line((cx, plot[1], cx, plot[3]), fill=(55, 62, 76), width=1)

        def point_for(r: dict) -> tuple[int, int]:
            px = cx + (float(r["time_vector_pca_x"]) / max_abs_x) * ((plot[2] - plot[0]) * 0.43)
            py = cy - (float(r["time_vector_pca_y"]) / max_abs_y) * ((plot[3] - plot[1]) * 0.43)
            return int(px), int(py)

        points = [point_for(r) for r in rows_sorted]
        for a, b in zip(points, points[1:]):
            draw.line((a[0], a[1], b[0], b[1]), fill=(115, 125, 145), width=2)
        for r, p in zip(rows_sorted, points):
            col = action_colors.get(r["emitted_action"], (220, 220, 220))
            radius = 5 if r is not row else 9
            draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=col, outline=(248, 249, 251) if r is row else None)
        draw.text((1148, 358), "PCA-1/2", fill=(178, 188, 204), font=font_small)
        draw.text((870, 532), f"time_vec_norm={row['time_vector_norm']:.2f}  hidden_norm={row['context_norm']:.2f}", fill=(230, 233, 238), font=font_small)
        draw.text((870, 556), f"cos(time, hidden)={row['time_context_cosine']:.3f}  step_L2={row['time_vector_step_l2']:.2f}", fill=(230, 233, 238), font=font_small)

        tx0, ty0, tx1, ty1 = 50, 620, 1230, 648
        draw.rectangle((tx0, ty0, tx1, ty1), fill=(45, 51, 62), outline=(110, 119, 134))
        speech_x = tx0 + int((speech_duration / duration) * (tx1 - tx0))
        draw.rectangle((tx0, ty0, speech_x, ty1), fill=(65, 145, 170))
        for r in rows_sorted:
            x = tx0 + int(((speech_duration + r["timepoint_s"]) / duration) * (tx1 - tx0))
            col = (248, 249, 251)
            if r["emitted_action"] == "BACKCHANNEL":
                col = (252, 141, 98)
            elif r["emitted_action"] == "SUPPORT":
                col = (141, 160, 203)
            draw.line((x, ty0 - 12, x, ty1 + 12), fill=col, width=2)
        now_x = tx0 + int((now / duration) * (tx1 - tx0))
        draw.line((now_x, ty0 - 28, now_x, ty1 + 28), fill=(255, 255, 255), width=3)
        draw.text((50, 656), "blue=user audio, ticks=decision loop, orange=BACKCHANNEL, purple=SUPPORT", fill=(178, 188, 204), font=font_small)

        writer.append_data(np.asarray(img))
    writer.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio_wav = out_mp4.with_suffix(".wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(out_video_noaudio),
        "-i",
        str(audio_wav),
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_rows_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "context_id",
        "profile",
        "timepoint_s",
        "gold",
        "raw_label",
        "emitted_action",
        "policy_reason",
        "p_WAIT",
        "p_BACKCHANNEL",
        "p_SUPPORT",
        "response_text",
        "hidden_mode",
        "layer",
        "context_norm",
        "time_vector_norm",
        "time_context_cosine",
        "time_vector_step_l2",
        "time_vector_pca_x",
        "time_vector_pca_y",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_demo(context_id: str, hidden_mode: str, interrupt_support_after: float | None = None, force_tts: bool = False):
    start = time.perf_counter()
    out_dir = OUT_DIR / f"{context_id}_{hidden_mode}" / ("interrupt" if interrupt_support_after is not None else "normal")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, rows = predict_context(context_id, hidden_mode)
    response_paths = synthesize_responses(out_dir / "response_tts", force=force_tts)
    wav_path = out_dir / "demo.wav"
    audio_meta = mix_audio(meta, rows, wav_path, response_paths, interrupt_support_after=interrupt_support_after)
    mp4_path = out_dir / "demo.mp4"
    noaudio_path = out_dir / "demo_noaudio.mp4"
    render_video(meta, rows, audio_meta, mp4_path, noaudio_path)
    write_rows_csv(out_dir / "decision_log.csv", rows)
    summary = {
        "meta": meta,
        "audio": audio_meta,
        "outputs": {
            "wav": str(wav_path),
            "mp4": str(mp4_path),
            "video_noaudio": str(noaudio_path),
            "decision_log": str(out_dir / "decision_log.csv"),
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - start,
        "notes": [
            "This demo uses cached Qwen2.5-Omni-3B Thinker hidden states from the prior v2 experiment.",
            "The frozen_0s mode reuses the same no-time context hidden and changes only the external Time Adapter features.",
            "Qwen3TTS is used for response audio when available; otherwise tone fallback is logged.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-id", default="test_neutral_incomplete_0000")
    parser.add_argument("--hidden-mode", choices=["frozen_0s", "recomputed_each_timepoint"], default="frozen_0s")
    parser.add_argument("--interrupt-support-after", type=float, default=None)
    parser.add_argument("--force-tts", action="store_true")
    args = parser.parse_args()
    summary = run_demo(args.context_id, args.hidden_mode, args.interrupt_support_after, args.force_tts)
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
