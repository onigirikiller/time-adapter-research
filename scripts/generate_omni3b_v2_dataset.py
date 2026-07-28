from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/omni3b_sequential_v2"
BASE_DIR = OUT_DIR / "base"
AUDIO_DIR = OUT_DIR / "audio"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
PROFILES = [
    "finished",
    "direct_question",
    "asked_wait",
    "self_repair",
    "neutral_incomplete",
    "vulnerable",
    "hesitant",
    "summary",
]
TIME_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
STAGES = {
    "small": {"train_contexts": 30, "validation_contexts": 10, "test_contexts": 10},
    "medium": {"train_contexts": 100, "validation_contexts": 20, "test_contexts": 20},
    "large": {"train_contexts": 300, "validation_contexts": 50, "test_contexts": 50},
    "extra": {"train_contexts": 500, "validation_contexts": 50, "test_contexts": 50},
}
SPLIT_CONTEXTS = {"train": 500, "validation": 50, "test": 50}
SEED = 20260623


TOPICS = [
    "the group project",
    "the message from my teacher",
    "the club meeting",
    "my homework plan",
    "the lunch conversation",
    "the exam schedule",
    "what my friend told me",
    "the family call",
    "the practice session",
    "the appointment tomorrow",
    "the mistake in my notes",
    "the plan after school",
    "the thing I promised",
    "the email I received",
    "the conversation yesterday",
    "the feedback from class",
    "the decision about Saturday",
    "the problem with my phone",
    "the story from the bus",
    "the plan for next week",
    "my answer to the question",
    "the awkward moment",
    "the schedule change",
    "the sentence I wrote",
    "the favor I need",
    "the important part",
    "the meeting I missed",
    "the idea I had",
    "the note from my parent",
    "the thing I forgot",
]

DETAILS = [
    "after I got home",
    "before the next class",
    "while everyone was listening",
    "when I checked it again",
    "because the timing felt wrong",
    "after thinking about it twice",
    "while I was trying to stay calm",
    "because I did not want to interrupt",
    "after the room got quiet",
    "when I noticed the difference",
    "because the wording matters",
    "while I was still deciding",
    "after I read the message again",
    "because I was not fully sure",
    "when the conversation stopped",
    "after the teacher left",
    "while I was looking for the right words",
    "because I wanted to be careful",
    "after everyone waited",
    "when I heard the question",
]

NAMES = [
    "Aki",
    "Mina",
    "Ren",
    "Yui",
    "Sora",
    "Nami",
    "Kai",
    "Emi",
    "Haru",
    "Riko",
    "Noa",
    "Toma",
]


TEMPLATES = {
    "finished": [
        "That is all I wanted to explain about {topic} {detail}.",
        "I think that covers {topic} {detail}.",
        "So that is my final answer about {topic} {detail}.",
        "That is the end of my explanation about {topic} {detail}.",
        "I have finished telling you about {topic} {detail}.",
    ],
    "direct_question": [
        "Could you help me decide what to do about {topic} {detail}?",
        "What do you think I should do about {topic} {detail}?",
        "Can you tell me how to respond to {topic} {detail}?",
        "What would you do with {topic} {detail}?",
        "Could you help me choose the next step for {topic} {detail}?",
    ],
    "asked_wait": [
        "Please give me a moment to think about {topic}; I am still choosing words.",
        "Do not rush me on {topic}; I am still putting it in order.",
        "Please wait while I sort out {topic} {detail}.",
        "I need a little more time before answering about {topic}.",
        "Give me a second, I am still thinking through {topic}.",
    ],
    "self_repair": [
        "Wait, that is not what I meant about {topic}; let me rephrase it.",
        "No, I should correct the part about {topic} {detail}.",
        "Actually I need to say {topic} differently.",
        "Hold on, I phrased {topic} wrong and need to fix it.",
        "Let me restart that sentence about {topic} because it came out wrong.",
    ],
    "neutral_incomplete": [
        "Today at school, {name} and I were talking about {topic} and",
        "The thing from yesterday about {topic} was that",
        "When I got home and checked {topic}, I noticed that",
        "There is something about {topic} {detail} that",
        "The reason I mentioned {topic} is because",
    ],
    "vulnerable": [
        "Honestly, I am scared to say this about {topic} {detail}.",
        "I feel embarrassed saying this, but {topic} has been hard for me.",
        "This is difficult to admit, but I might need help with {topic}.",
        "I feel a little overwhelmed by {topic} {detail}.",
        "I am not sure how to say this, but {topic} is affecting me.",
    ],
    "hesitant": [
        "Um, I am not sure how to explain {topic} {detail}.",
        "This is a little hard to say, but {topic}",
        "I am trying to find the right words for {topic} {detail}.",
        "I started saying it, but I got stuck on {topic}.",
        "Maybe the problem with {topic} is, uh,",
    ],
    "summary": [
        "So the main point about {topic} is",
        "To summarize what I mean about {topic},",
        "The conclusion from {topic} {detail} is probably",
        "What I really want to say about {topic} is",
        "If I put {topic} simply, then",
    ],
}


def target_label(profile: str, silence: float) -> tuple[str, list[str]]:
    if profile == "asked_wait":
        return "WAIT", ["WAIT", "BACKCHANNEL"] if silence >= 4.0 else ["WAIT"]
    if profile in {"finished", "direct_question"}:
        return "SUPPORT", ["SUPPORT"]
    if profile == "self_repair":
        if silence < 1.0:
            return "WAIT", ["WAIT"]
        return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
    if profile == "vulnerable":
        if silence < 0.5:
            return "BACKCHANNEL", ["BACKCHANNEL", "SUPPORT"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    if profile == "neutral_incomplete":
        if silence < 0.75:
            return "WAIT", ["WAIT"]
        if silence < 3.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    if profile == "hesitant":
        if silence < 0.5:
            return "WAIT", ["WAIT"]
        if silence < 3.0:
            return "BACKCHANNEL", ["BACKCHANNEL"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    if profile == "summary":
        if silence < 0.75:
            return "WAIT", ["WAIT"]
        if silence < 4.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"]
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"]
    raise ValueError(profile)


def make_features(silence: float, previous: float | None, speech_duration: float):
    delta_t = 0.0 if previous is None else round(silence - previous, 4)
    return {
        "silence_elapsed": silence,
        "delta_t": delta_t,
        "utterance_elapsed": round(speech_duration + silence, 4),
        "is_user_speaking": silence == 0.0,
        "asr_changed": silence == 0.0,
    }


def init_tts():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation="eager",
    )


def synthesize_base_batch(model, items: list[dict]):
    pending = [item for item in items if not item["base_path"].exists() or item["base_path"].stat().st_size <= 4096]
    if not pending:
        return
    texts = [item["fragment"] for item in pending]
    instructs = []
    for item in pending:
        profile = item["profile"]
        if profile in {"finished", "direct_question"}:
            style = "Speak naturally and clearly, as a completed conversational turn."
        elif profile in {"asked_wait", "self_repair"}:
            style = "Speak naturally, with a thinking or self-correcting tone, without adding a long pause."
        elif profile in {"vulnerable", "hesitant"}:
            style = "Speak gently and naturally, with a hesitant conversational tone, without adding a long pause."
        else:
            style = "Speak naturally as an unfinished dialogue fragment, without adding a long pause."
        instructs.append(style)
    wavs, sr = model.generate_custom_voice(
        text=texts,
        language=["English"] * len(texts),
        speaker=["Ryan"] * len(texts),
        instruct=instructs,
        max_new_tokens=300,
    )
    for item, wav in zip(pending, wavs):
        trimmed = trim_wav_array(np.asarray(wav, dtype=np.float32), sr)
        sf.write(str(item["base_path"]), trimmed, sr)


def trim_wav_array(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    trimmed, _ = librosa.effects.trim(wav, top_db=35)
    if trimmed.size >= int(sr * 0.15):
        wav = trimmed.astype(np.float32)
    return wav.astype(np.float32)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def render_fragment(profile: str, idx: int, split: str) -> str:
    template = TEMPLATES[profile][idx % len(TEMPLATES[profile])]
    topic = TOPICS[(idx * 7 + len(split) * 3 + len(profile)) % len(TOPICS)]
    detail = DETAILS[(idx * 11 + len(profile) * 5 + len(split)) % len(DETAILS)]
    name = NAMES[(idx * 13 + len(split)) % len(NAMES)]
    marker = f"case {split} {profile.replace('_', ' ')} {idx:03d}"
    text = template.format(topic=topic, detail=detail, name=name)
    return f"{text} ({marker})"


def round_robin_contexts(split: str, n_contexts: int) -> list[dict]:
    counters = {profile: 0 for profile in PROFILES}
    contexts = []
    for global_idx in range(n_contexts):
        profile = PROFILES[global_idx % len(PROFILES)]
        local_idx = counters[profile]
        counters[profile] += 1
        context_id = f"{split}_{profile}_{local_idx:04d}"
        fragment = render_fragment(profile, local_idx, split)
        contexts.append(
            {
                "context_id": context_id,
                "split": split,
                "split_context_index": global_idx,
                "profile": profile,
                "fragment": fragment,
            }
        )
    return contexts


def stage_context_limit(stage: str, split: str) -> int:
    return STAGES[stage][f"{split}_contexts"]


def assign_stages(row: dict) -> list[str]:
    stages = []
    idx = row["split_context_index"]
    for stage in STAGES:
        if idx < stage_context_limit(stage, row["split"]):
            stages.append(stage)
    return stages


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    model = init_tts()

    contexts = []
    for split, n in SPLIT_CONTEXTS.items():
        contexts.extend(round_robin_contexts(split, n))

    base_plan = [{**ctx, "base_path": BASE_DIR / f"{ctx['context_id']}.wav"} for ctx in contexts]
    batch_size = 4
    for start in range(0, len(base_plan), batch_size):
        batch = base_plan[start : start + batch_size]
        synthesize_base_batch(model, batch)
        print(f"Qwen3TTS generated base contexts {min(start + batch_size, len(base_plan))}/{len(base_plan)}", flush=True)

    rows = []
    base_rows = []
    for i, ctx in enumerate(contexts, start=1):
        base_path = BASE_DIR / f"{ctx['context_id']}.wav"
        wav, sr = read_wav(base_path)
        trimmed_path = base_path
        speech_duration = float(len(wav) / sr)
        base = {
            **ctx,
            "sample_rate": sr,
            "speech_duration_seconds": round(speech_duration, 4),
            "base_audio_path": str(base_path),
            "trimmed_audio_path": str(trimmed_path),
            "stages": assign_stages(ctx),
        }
        base_rows.append(base)
        previous = None
        for silence in TIME_POINTS:
            silence_wav = np.zeros(int(round(sr * silence)), dtype=np.float32)
            combined = np.concatenate([wav, silence_wav]).astype(np.float32)
            out_path = AUDIO_DIR / f"{ctx['context_id']}_silence_{silence:g}s.wav"
            if not out_path.exists() or out_path.stat().st_size <= 4096:
                sf.write(str(out_path), combined, sr)
            label, acceptable = target_label(ctx["profile"], silence)
            row = {
                "id": f"{ctx['context_id']}_{silence:g}s",
                **ctx,
                "stages": assign_stages(ctx),
                "silence_seconds": silence,
                "features": make_features(silence, previous, speech_duration),
                "label": label,
                "acceptable_labels": acceptable,
                "sample_rate": sr,
                "speech_duration_seconds": round(speech_duration, 4),
                "total_duration_seconds": round(float(len(combined) / sr), 4),
                "audio_path": str(out_path),
            }
            rows.append(row)
            previous = silence
        if i % 50 == 0:
            print(f"Prepared silence variants {i}/{len(contexts)} contexts", flush=True)

    write_jsonl(OUT_DIR / "contexts.jsonl", base_rows)
    for split in ["train", "validation", "test"]:
        write_jsonl(OUT_DIR / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    for stage in STAGES:
        stage_dir = OUT_DIR / stage
        stage_dir.mkdir(exist_ok=True)
        for split in ["train", "validation", "test"]:
            write_jsonl(stage_dir / f"{split}.jsonl", [row for row in rows if row["split"] == split and stage in row["stages"]])

    text_by_split = defaultdict(set)
    for ctx in base_rows:
        text_by_split[ctx["split"]].add(ctx["fragment"])
    overlaps = {}
    for a in text_by_split:
        for b in text_by_split:
            if a < b:
                overlaps[f"{a}_{b}"] = len(text_by_split[a] & text_by_split[b])

    stage_summary = {}
    for stage in STAGES:
        stage_rows = [row for row in rows if stage in row["stages"]]
        stage_summary[stage] = {
            "split_counts": dict(Counter(row["split"] for row in stage_rows)),
            "label_counts": {
                split: dict(Counter(row["label"] for row in stage_rows if row["split"] == split))
                for split in ["train", "validation", "test"]
            },
            "profile_counts": {
                split: dict(Counter(row["profile"] for row in stage_rows if row["split"] == split))
                for split in ["train", "validation", "test"]
            },
            "context_counts": {
                split: len({row["context_id"] for row in stage_rows if row["split"] == split})
                for split in ["train", "validation", "test"]
            },
        }

    same_second_labels = {
        str(t): sorted({row["label"] for row in rows if row["silence_seconds"] == t})
        for t in TIME_POINTS
    }
    profile_label_by_time = {
        profile: {str(t): target_label(profile, t)[0] for t in TIME_POINTS}
        for profile in PROFILES
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "model_target": "Qwen/Qwen2.5-Omni-3B",
        "profiles": PROFILES,
        "time_points": TIME_POINTS,
        "stages": STAGES,
        "max_split_counts": dict(Counter(row["split"] for row in rows)),
        "max_context_counts": SPLIT_CONTEXTS,
        "stage_summary": stage_summary,
        "same_second_labels": same_second_labels,
        "profile_label_by_time": profile_label_by_time,
        "split_fragment_overlap": overlaps,
        "feature_names": ["silence_elapsed", "delta_t", "utterance_elapsed", "is_user_speaking", "asr_changed"],
        "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice on local GPU when CUDA is available, then waveform trim.",
        "waveform_silence": "Zero-valued PCM samples appended after the trimmed base utterance.",
        "hard_negative_design": [
            "finished/direct_question remain SUPPORT even at 0.0s and 0.25s.",
            "asked_wait remains WAIT through 6.0s to prevent automatic escalation.",
            "self_repair is WAIT before 1.0s and BACKCHANNEL afterward, never SUPPORT.",
            "vulnerable escalates to SUPPORT after 0.5s.",
            "neutral_incomplete transitions WAIT -> BACKCHANNEL -> SUPPORT.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
