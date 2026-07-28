import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


FRAGMENTS = [
    "Today at school I",
    "I wanted to tell you that",
    "Something awkward happened when",
]
SILENCES = [0.0, 0.5, 1.0, 4.0]


def trim_audio(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    trimmed, _ = librosa.effects.trim(wav, top_db=35)
    if trimmed.size < 2400:
        return wav
    return trimmed.astype(np.float32)


def main():
    out_dir = Path("data/audio_silence_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    base_dir = out_dir / "base"
    base_dir.mkdir(exist_ok=True)

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )

    rows = []
    for fragment_index, fragment in enumerate(FRAGMENTS):
        base_path = base_dir / f"fragment_{fragment_index:02d}.wav"
        if base_path.exists():
            wav, sr = sf.read(str(base_path), dtype="float32")
        else:
            wavs, sr = model.generate_custom_voice(
                text=fragment,
                language="English",
                speaker="Ryan",
                instruct="Natural conversational voice.",
                max_new_tokens=260,
            )
            wav = trim_audio(wavs[0])
            sf.write(str(base_path), wav, sr)

        for silence_seconds in SILENCES:
            silence = np.zeros(int(sr * silence_seconds), dtype=np.float32)
            combined = np.concatenate([np.asarray(wav, dtype=np.float32).reshape(-1), silence])
            out_path = out_dir / f"fragment_{fragment_index:02d}_silence_{silence_seconds:.1f}s.wav"
            sf.write(str(out_path), combined, sr)
            rows.append(
                {
                    "fragment_index": fragment_index,
                    "fragment": fragment,
                    "silence_seconds": silence_seconds,
                    "sample_rate": sr,
                    "speech_duration_seconds": round(float(len(wav) / sr), 4),
                    "total_duration_seconds": round(float(len(combined) / sr), 4),
                    "audio_path": str(out_path),
                }
            )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
