import json
import re
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]


def parse_label(text: str) -> str:
    upper = text.upper()
    for label in LABELS:
        if re.search(rf"\b{label}\b", upper):
            return label
    return "UNPARSED"


def build_conversation(audio_path: Path):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a timing classifier for a streaming spoken dialogue system. "
                        "Listen to the user audio and classify what the assistant should do."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path.resolve())},
                {
                    "type": "text",
                    "text": (
                        "The audio contains a partial user utterance followed by possible silence. "
                        "Choose exactly one label and output only that label: "
                        "WAIT for no/short silence, BACKCHANNEL for a moderate pause, "
                        "SUPPORT for a long pause."
                    ),
                },
            ],
        },
    ]


def main():
    manifest_path = Path("data/audio_silence_dataset/manifest.jsonl")
    output_path = Path("artifacts/omni_audio_timing/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    model_id = "Qwen/Qwen2.5-Omni-3B"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir="./.cache/huggingface",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_id,
        cache_dir="./.cache/huggingface",
    )

    results = []
    for row in rows:
        conversation = build_conversation(Path(row["audio_path"]))
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            text_ids = model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=24,
                do_sample=False,
            )
        new_text_ids = text_ids[:, input_length:]
        decoded = processor.batch_decode(
            new_text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        result = dict(row)
        result["raw_output"] = decoded
        result["parsed_label"] = parse_label(decoded)
        results.append(result)
        print(row["audio_path"], row["silence_seconds"], result["parsed_label"], repr(decoded[:200]))

    summary = {
        "model_id": model_id,
        "num_examples": len(results),
        "results": results,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
