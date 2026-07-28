import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]


def target_label(silence_seconds: float) -> str:
    if silence_seconds < 1.0:
        return "WAIT"
    if silence_seconds < 4.0:
        return "BACKCHANNEL"
    return "SUPPORT"


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
                        "Choose exactly one label and output only that label: WAIT, BACKCHANNEL, or SUPPORT."
                    ),
                },
            ],
        },
    ]


def move_inputs(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def make_features(processor, row, label=None):
    conversation = build_conversation(Path(row["audio_path"]))
    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = prompt_text if label is None else prompt_text + label
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=full_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    if label is None:
        return inputs, inputs["input_ids"].shape[1]
    prompt_inputs = processor(
        text=prompt_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return inputs, prompt_inputs["input_ids"].shape[1]


def evaluate(model, processor, rows, device, dtype):
    model.eval()
    results = []
    for row in rows:
        inputs, input_length = make_features(processor, row, label=None)
        inputs = move_inputs(inputs, device, dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=12,
                do_sample=False,
            )
        decoded = processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        result = dict(row)
        result["target_label"] = target_label(float(row["silence_seconds"]))
        result["raw_output"] = decoded
        result["parsed_label"] = parse_label(decoded)
        results.append(result)
    return results


def main():
    manifest_path = Path("data/audio_silence_dataset/manifest.jsonl")
    output_dir = Path("artifacts/omni_audio_timing_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    model_id = "Qwen/Qwen2.5-Omni-3B"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir="./.cache/huggingface",
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.disable_talker()
    model.config.use_cache = False
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_id,
        cache_dir="./.cache/huggingface",
    )
    device = model.device

    before = evaluate(model, processor, rows, device, dtype)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.thinker.parameters() if p.requires_grad], lr=2e-4)
    history = []
    for epoch in range(8):
        model.train()
        total_loss = 0.0
        for row in rows:
            label = target_label(float(row["silence_seconds"]))
            inputs, prompt_len = make_features(processor, row, label=label)
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            inputs = move_inputs(inputs, device, dtype)
            labels = labels.to(device)
            outputs = model.thinker(**inputs, labels=labels, use_audio_in_video=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu())
        avg_loss = total_loss / len(rows)
        history.append({"epoch": epoch, "avg_loss": avg_loss})
        print(f"epoch={epoch} avg_loss={avg_loss:.4f}")

    after = evaluate(model, processor, rows, device, dtype)
    model.thinker.save_pretrained(output_dir / "adapter")
    summary = {
        "model_id": model_id,
        "num_examples": len(rows),
        "training_history": history,
        "before": before,
        "after": after,
        "adapter_path": str(output_dir / "adapter"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
