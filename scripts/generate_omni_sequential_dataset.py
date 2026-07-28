import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/omni_sequential_time_adapter"
BASE_DIR = OUT_DIR / "base"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
TIME_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


FRAGMENTS = {
    "neutral_incomplete": [
        "Today at school I",
        "The thing from the meeting was",
        "When I got home I noticed",
        "There is something from yesterday that",
    ],
    "asked_wait": [
        "Give me a moment to think",
        "Please wait while I put this in order",
        "I need a second to choose the words",
        "Do not rush me, I am still thinking",
    ],
    "finished": [
        "That is all I wanted to say",
        "That is the end of my explanation",
        "So that is my answer",
        "I think that covers everything",
    ],
    "hesitant": [
        "Um, I am not sure how to say this",
        "This is a little hard to explain",
        "I am trying to find the right words",
        "I started saying it but I got stuck",
    ],
    "summary": [
        "So the main point is",
        "To summarize what I mean",
        "The conclusion is probably",
        "What I really want to say is",
    ],
    "vulnerable": [
        "Sorry, this is hard to say",
        "Honestly I am a little scared to say it",
        "I feel embarrassed saying this but",
        "I think I might need help with this",
    ],
    "self_repair": [
        "Wait, that is not what I meant",
        "No, let me say that differently",
        "Actually I should correct that",
        "Hold on, I need to rephrase it",
    ],
    "direct_question": [
        "What do you think I should do",
        "Can you tell me how to respond",
        "What would you do in that situation",
        "Could you help me decide what comes next",
    ],
}


def target_label(profile: str, silence: float) -> tuple[str, list[str]]:
    if profile == "asked_wait":
        if silence >= 6.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
        return "WAIT", ["WAIT"]
    if profile in {"finished", "direct_question"}:
        return "SUPPORT", ["SUPPORT"]
    if profile == "vulnerable":
        if silence < 0.5:
            return "BACKCHANNEL", ["BACKCHANNEL", "SUPPORT"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    if profile == "self_repair":
        if silence < 1.0:
            return "WAIT", ["WAIT"]
        return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
    if profile == "summary":
        if silence < 0.75:
            return "WAIT", ["WAIT"]
        if silence < 4.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    if profile == "hesitant":
        if silence < 0.5:
            return "WAIT", ["WAIT"]
        if silence < 3.0:
            return "BACKCHANNEL", ["BACKCHANNEL"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    if profile == "neutral_incomplete":
        if silence < 0.75:
            return "WAIT", ["WAIT"]
        if silence < 3.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    raise ValueError(profile)


def trim_audio(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    trimmed, _ = librosa.effects.trim(wav, top_db=35)
    if trimmed.size < 2400:
        return wav
    return trimmed.astype(np.float32)


def split_for_variant(variant: int) -> str:
    if variant < 2:
        return "train"
    if variant == 2:
        return "validation"
    return "test"


def make_features(silence: float, previous: float | None, speech_duration: float):
    delta_t = 0.0 if previous is None else round(silence - previous, 4)
    return {
        "silence_elapsed": silence,
        "delta_t": delta_t,
        "utterance_elapsed": round(speech_duration + silence, 4),
        "is_user_speaking": silence == 0.0,
        "asr_changed": silence == 0.0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation="eager",
    )

    rows = []
    base_rows = []
    for profile, fragments in FRAGMENTS.items():
        for variant, fragment in enumerate(fragments):
            split = split_for_variant(variant)
            context_id = f"{profile}_{variant}"
            base_path = BASE_DIR / f"{context_id}.wav"
            if base_path.exists():
                wav, sr = sf.read(str(base_path), dtype="float32")
            else:
                wavs, sr = model.generate_custom_voice(
                    text=fragment,
                    language="English",
                    speaker="Ryan",
                    instruct="Natural conversational voice. Speak the phrase as an unfinished dialogue fragment.",
                    max_new_tokens=260,
                )
                wav = trim_audio(wavs[0])
                sf.write(str(base_path), wav, sr)
            speech_duration = float(len(wav) / sr)
            base_rows.append(
                {
                    "context_id": context_id,
                    "profile": profile,
                    "split": split,
                    "fragment": fragment,
                    "sample_rate": sr,
                    "speech_duration_seconds": round(speech_duration, 4),
                    "base_audio_path": str(base_path),
                }
            )
            previous = None
            for silence in TIME_POINTS:
                silence_wav = np.zeros(int(round(sr * silence)), dtype=np.float32)
                combined = np.concatenate([np.asarray(wav, dtype=np.float32).reshape(-1), silence_wav])
                out_path = OUT_DIR / f"{context_id}_silence_{silence:g}s.wav"
                sf.write(str(out_path), combined, sr)
                label, acceptable = target_label(profile, silence)
                rows.append(
                    {
                        "id": f"{context_id}_{silence:g}s",
                        "split": split,
                        "context_id": context_id,
                        "profile": profile,
                        "fragment": fragment,
                        "silence_seconds": silence,
                        "features": make_features(silence, previous, speech_duration),
                        "label": label,
                        "acceptable_labels": acceptable,
                        "sample_rate": sr,
                        "speech_duration_seconds": round(speech_duration, 4),
                        "total_duration_seconds": round(float(len(combined) / sr), 4),
                        "audio_path": str(out_path),
                    }
                )
                previous = silence

    for split in ["train", "validation", "test"]:
        with (OUT_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                if row["split"] == split:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT_DIR / "base_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in base_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profiles": sorted(FRAGMENTS),
        "time_points": TIME_POINTS,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "label_counts": {split: dict(Counter(row["label"] for row in rows if row["split"] == split)) for split in ["train", "validation", "test"]},
        "profile_counts": {split: dict(Counter(row["profile"] for row in rows if row["split"] == split)) for split in ["train", "validation", "test"]},
        "waveform_silence": "Silence is appended as zero-valued PCM samples after trimming generated base speech.",
        "feature_names": ["silence_elapsed", "delta_t", "utterance_elapsed", "is_user_speaking", "asr_changed"],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
