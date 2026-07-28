from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/omni3b_control_response_lora_v1"
BASE_LABEL_LORA = ROOT / "artifacts/omni3b_single_token_lora_v1/clean0p6b_large_slash_l3_a4_audio_only_1000pc_e2_fulltest/best_lora"
SEED = 20260628


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


single = import_module(ROOT / "scripts/run_omni3b_single_token_lora_v1.py", "single_token_core_response_lora")
single.DATA_DIR = ROOT / "data/omni3b_delayed_backchannel_v2_qwen3tts_clean_0p6b"
single.SOURCE_DATA_DIR = ROOT / "data/omni3b_sequential_v2"

LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
CODEBOOK = {"WAIT": "/W", "BACKCHANNEL": "/B", "SUPPORT": "/S"}

BACKCHANNEL_TEXT = {
    "asked_wait": ["Take your time.", "Mm-hm.", "I'm listening."],
    "self_repair": ["Mm-hm.", "Yeah.", "Go on."],
    "neutral_incomplete": ["Mm-hm.", "Yeah.", "I'm listening."],
    "vulnerable": ["I'm listening.", "Mm-hm.", "Yeah."],
    "hesitant": ["Take your time.", "Mm-hm.", "I'm listening."],
    "summary": ["Mm-hm.", "Yeah.", "I hear you."],
    "finished": ["Got it.", "Okay.", "I see."],
    "direct_question": ["Okay.", "I see.", "Got it."],
}

SUPPORT_TEXT = {
    "asked_wait": ["Take your time. I'm listening.", "No rush. I'm here."],
    "self_repair": ["Take your time. You can continue.", "No rush. Keep going."],
    "neutral_incomplete": ["You can keep going.", "I'm here with you."],
    "vulnerable": ["I'm here with you.", "That sounds really hard."],
    "hesitant": ["Take your time. I'm here.", "You can say it slowly."],
    "summary": ["That makes sense.", "I understand the main point."],
    "finished": ["Thanks for explaining that.", "I understand. Thank you."],
    "direct_question": ["I can help with that.", "Let's work through it."],
}

BANNED_RESPONSE_BITS = ["WAIT", "BACKCHANNEL", "SUPPORT", "label", "timer", "system", "version"]


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


def response_for_row(row: dict) -> str:
    label = row["label"]
    profile = row["profile"]
    if label == "WAIT":
        return CODEBOOK["WAIT"]
    pool = BACKCHANNEL_TEXT.get(profile, BACKCHANNEL_TEXT["neutral_incomplete"]) if label == "BACKCHANNEL" else SUPPORT_TEXT.get(profile, SUPPORT_TEXT["neutral_incomplete"])
    key = f"{row['id']}|{profile}|{label}"
    idx = sum(ord(ch) for ch in key) % len(pool)
    return CODEBOOK[label] + " " + pool[idx]


def make_input_with_target(processor, row: dict, target: str, eos: bool = True):
    conv = single.build_conversation(row, "audio_only", CODEBOOK, explicit_seconds=False, audio_timing_mode="row_audio")
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    target_text = target
    eos_token = processor.tokenizer.eos_token or ""
    if eos and eos_token and not target_text.endswith(eos_token):
        target_text += eos_token
    inputs = processor(
        text=prompt_text + target_text,
        audio=single.process_mm_info(conv, use_audio_in_video=False)[0],
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    prompt_inputs = processor(
        text=prompt_text,
        audio=single.process_mm_info(conv, use_audio_in_video=False)[0],
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return inputs, int(prompt_inputs["input_ids"].shape[1]), prompt_text, target_text


def make_prompt_inputs(processor, row: dict):
    conv = single.build_conversation(row, "audio_only", CODEBOOK, explicit_seconds=False, audio_timing_mode="row_audio")
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = single.process_mm_info(conv, use_audio_in_video=False)
    inputs = processor(text=prompt_text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    return inputs, int(inputs["input_ids"].shape[1]), prompt_text


def attach_trainable_lora(model, adapter_path: Path):
    model.thinker = PeftModel.from_pretrained(model.thinker, adapter_path, is_trainable=True)
    return model


def weighted_lm_loss(logits: torch.Tensor, labels: torch.Tensor, prompt_len: int, control_weight: float):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    raw = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    weights = torch.ones_like(shift_labels, dtype=logits.dtype)
    first_target_shift_index = max(0, prompt_len - 1)
    if first_target_shift_index < weights.shape[1]:
        weights[:, first_target_shift_index] = control_weight
    denom = (weights * mask).sum().clamp_min(1.0)
    return (raw * weights * mask).sum() / denom


def train_one_epoch(model, processor, rows, adapter, optimizer, device, dtype, args, epoch: int, log_path: Path):
    model.thinker.train()
    order = list(range(len(rows)))
    random.Random(args.seed + epoch).shuffle(order)
    total = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()
    for idx in order:
        row = rows[idx]
        target = response_for_row(row)
        inputs, prompt_len, _, target_text = make_input_with_target(processor, row, target, eos=True)
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        moved = single.move_inputs(inputs, device, dtype)
        labels = labels.to(device)
        vector = single.adapter_predict(adapter, [row])[0]
        hook = single.InjectionHook(vector, alpha=args.alpha, position=args.position, enabled=True)
        handle = single.hook_module(model, args.layer).register_forward_hook(hook)
        try:
            outputs = model.thinker(**moved, use_audio_in_video=False)
            loss = weighted_lm_loss(outputs.logits, labels, prompt_len, args.control_loss_weight) / args.grad_accum
            loss.backward()
        finally:
            handle.remove()
        total += float(loss.detach().cpu()) * args.grad_accum
        steps += 1
        if steps % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.thinker.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available() and steps % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        if steps == 1 or steps % args.log_every == 0 or steps == len(rows):
            msg = {
                "epoch": epoch,
                "step": steps,
                "rows": len(rows),
                "avg_loss": total / max(steps, 1),
                "elapsed_sec": time.time() - start,
                "last_target": target_text,
                "hook_calls": hook.calls,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            print(f"epoch={epoch} step={steps}/{len(rows)} loss={msg['avg_loss']:.4f}", flush=True)
    if steps % args.grad_accum:
        torch.nn.utils.clip_grad_norm_([p for p in model.thinker.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"epoch": epoch, "train_loss": total / max(steps, 1), "train_steps": steps}


def parse_control_text(text: str) -> tuple[str, str]:
    stripped = text.strip()
    for label, token in CODEBOOK.items():
        if stripped.startswith(token):
            return label, stripped[len(token):].strip()
    m = re.search(r"(/W|/B|/S)", stripped)
    if m:
        token = m.group(1)
        label = {v: k for k, v in CODEBOOK.items()}[token]
        return label, stripped[m.end():].strip()
    return "PARSE_FAIL", stripped


def response_quality(label: str, text: str) -> dict:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    banned = any(bit.lower() in text.lower() for bit in BANNED_RESPONSE_BITS)
    has_question = "?" in text
    if label == "WAIT":
        ok = len(text.strip()) == 0
    elif label == "BACKCHANNEL":
        ok = 0 < len(words) <= 4 and not has_question and not banned
    elif label == "SUPPORT":
        ok = 2 <= len(words) <= 14 and not banned
    else:
        ok = False
    return {
        "response_words": len(words),
        "response_has_question": int(has_question),
        "response_has_banned_text": int(banned),
        "response_quality_ok": int(ok),
    }


def eval_label_logits(model, processor, rows, split: str, vectors, device, dtype, args, token_ids, conditions):
    model.thinker.eval()
    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            prompt_inputs, prompt_len, _ = make_prompt_inputs(processor, row)
            moved = single.move_inputs(prompt_inputs, device, dtype)
            for cond in conditions:
                vector = vectors[cond][i]
                hook = single.InjectionHook(vector, alpha=args.alpha, position=args.position, enabled=(cond != "no_time"))
                handle = single.hook_module(model, args.layer).register_forward_hook(hook)
                try:
                    t0 = time.perf_counter()
                    outputs = model.thinker(**moved, use_audio_in_video=False)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                finally:
                    handle.remove()
                logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
                label_logits = np.array([logits[token_ids[label]] for label in LABELS], dtype=np.float64)
                probs = single.softmax3(label_logits)
                pred = LABELS[int(np.argmax(probs))]
                results.append(
                    {
                        "id": row["id"],
                        "context_id": row["context_id"],
                        "split": split,
                        "profile": row["profile"],
                        "fragment": row["fragment"],
                        "silence_seconds": row["silence_seconds"],
                        "gold_label": row["label"],
                        "target_response": response_for_row(row),
                        "condition": cond,
                        "pred_label": pred,
                        "correct": int(pred == row["label"]),
                        "wait_prob": float(probs[0]),
                        "backchannel_prob": float(probs[1]),
                        "support_prob": float(probs[2]),
                        "wait_logit": float(label_logits[0]),
                        "backchannel_logit": float(label_logits[1]),
                        "support_logit": float(label_logits[2]),
                        "latency_ms": latency_ms,
                    }
                )
            if (i + 1) % args.eval_log_every == 0 or i + 1 == len(rows):
                print(f"logit eval {split}: {i + 1}/{len(rows)}", flush=True)
    return results


def generate_rows(model, processor, rows, split: str, vectors, device, dtype, args, limit: int):
    selected = rows[:limit] if limit else rows
    model.thinker.eval()
    out = []
    with torch.inference_mode():
        for i, row in enumerate(selected):
            prompt_inputs, prompt_len, _ = make_prompt_inputs(processor, row)
            moved = single.move_inputs(prompt_inputs, device, dtype)
            vector = vectors["correct_time_adapter"][i]
            hook = single.InjectionHook(vector, alpha=args.alpha, position=args.position, enabled=True)
            handle = single.hook_module(model, args.layer).register_forward_hook(hook)
            try:
                gen = model.thinker.generate(
                    **moved,
                    max_new_tokens=args.gen_max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                    use_audio_in_video=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            finally:
                handle.remove()
            generated_ids = gen[0, prompt_len:].detach().cpu().tolist()
            text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred_label, response_text = parse_control_text(text)
            quality = response_quality(pred_label, response_text)
            out.append(
                {
                    "id": row["id"],
                    "context_id": row["context_id"],
                    "split": split,
                    "profile": row["profile"],
                    "silence_seconds": row["silence_seconds"],
                    "gold_label": row["label"],
                    "target_response": response_for_row(row),
                    "generated_raw": text,
                    "generated_label": pred_label,
                    "generated_response": response_text,
                    "label_correct": int(pred_label == row["label"]),
                    **quality,
                }
            )
            if (i + 1) % args.gen_log_every == 0 or i + 1 == len(selected):
                print(f"generate eval {split}: {i + 1}/{len(selected)}", flush=True)
    return out


def classification_metrics(rows: list[dict]) -> dict:
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    out = {}
    idx = {label: i for i, label in enumerate(LABELS)}
    for cond, items in by_cond.items():
        y = [idx[r["gold_label"]] for r in items]
        p = [idx[r["pred_label"]] for r in items]
        out[cond] = {
            "accuracy": float(np.mean([r["correct"] for r in items])),
            "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
            "classification_report": classification_report(y, p, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "pred_counts": dict(Counter(r["pred_label"] for r in items)),
        }
    return out


def generation_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "rows": len(rows),
        "label_accuracy": float(np.mean([r["label_correct"] for r in rows])),
        "response_quality_ok_rate": float(np.mean([r["response_quality_ok"] for r in rows])),
        "wait_extra_text_rate": float(np.mean([1 for r in rows if r["gold_label"] == "WAIT" and r["generated_response"].strip()] or [0])),
        "pred_counts": dict(Counter(r["generated_label"] for r in rows)),
        "by_gold": {
            label: {
                "rows": len(items),
                "label_accuracy": float(np.mean([r["label_correct"] for r in items])) if items else 0.0,
                "quality_ok_rate": float(np.mean([r["response_quality_ok"] for r in items])) if items else 0.0,
                "mean_response_words": float(np.mean([r["response_words"] for r in items])) if items else 0.0,
            }
            for label, items in ((lab, [r for r in rows if r["gold_label"] == lab]) for lab in LABELS)
        },
    }


def build_vectors(rows, split: str, args):
    return single.build_condition_vectors(rows, split, args.layer, args.seed)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device, dtype = single.load_model(args.dtype, args.attn, use_cache=False, gradient_checkpointing=args.gradient_checkpointing)
    token_ids = single.label_token_ids(processor.tokenizer, CODEBOOK)
    adapter, adapter_artifact = single.load_time_adapter(args.layer)

    train_all = single.load_rows("train")
    val_rows = single.limited_contexts(single.load_rows("validation"), args.val_contexts)
    test_rows = single.limited_contexts(single.load_rows("test"), args.test_contexts)
    train_rows = single.balanced_sample(train_all, args.train_per_class, args.seed, allow_oversample=True)
    config = {
        "run_name": args.run_name,
        "base_model": single.MODEL_ID,
        "base_label_lora": str(BASE_LABEL_LORA),
        "dataset": str(single.DATA_DIR),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "label_counts": {"train": single.label_counts(train_rows), "validation": single.label_counts(val_rows), "test": single.label_counts(test_rows)},
        "codebook": CODEBOOK,
        "token_ids": token_ids,
        "timepoints": sorted({float(r["silence_seconds"]) for r in train_all}),
        "target_policy": {"backchannel": BACKCHANNEL_TEXT, "support": SUPPORT_TEXT, "wait": "/W only"},
        "loss": {"control_loss_weight": args.control_loss_weight, "full_response_loss": True, "eos_trained": True},
        "layer": args.layer,
        "alpha": args.alpha,
        "position": args.position,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    write_json(run_dir / "config.json", config)

    model = attach_trainable_lora(model, BASE_LABEL_LORA)
    params = single.trainable_parameter_summary(model)
    write_json(run_dir / "trainable_parameters.json", params)
    print(json.dumps({"config": config, "trainable": params}, ensure_ascii=False), flush=True)

    optimizer = torch.optim.AdamW([p for p in model.thinker.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        hist = train_one_epoch(model, processor, train_rows, adapter, optimizer, device, dtype, args, epoch, run_dir / "train_log.jsonl")
        val_vectors = build_vectors(val_rows, "validation", args)
        val_results = eval_label_logits(model, processor, val_rows, "validation", val_vectors, device, dtype, args, token_ids, ["correct_time_adapter"])
        val_metrics = classification_metrics(val_results)
        score = val_metrics["correct_time_adapter"]["macro_f1"]
        hist["validation_macro_f1"] = score
        hist["validation_accuracy"] = val_metrics["correct_time_adapter"]["accuracy"]
        history.append(hist)
        write_json(run_dir / "training_history.json", history)
        write_json(run_dir / f"epoch_{epoch:02d}_validation_metrics.json", val_metrics)
        if score >= best_score:
            best_score = score
            best_dir = run_dir / "best_lora"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.thinker.save_pretrained(best_dir)
            write_json(run_dir / "best_checkpoint.json", {"epoch": epoch, "macro_f1": score, "path": str(best_dir)})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_conditions = [c for c in args.eval_conditions.split(",") if c]
    all_results = []
    for split, rows in [("validation", val_rows), ("test", test_rows)]:
        vectors = build_vectors(rows, split, args)
        results = eval_label_logits(model, processor, rows, split, vectors, device, dtype, args, token_ids, eval_conditions)
        all_results.extend(results)
        write_csv(run_dir / f"{split}_per_condition_results.csv", results)
        write_json(run_dir / f"{split}_metrics.json", classification_metrics(results))
        gen_rows = generate_rows(model, processor, rows, split, vectors, device, dtype, args, args.generation_eval_limit)
        write_csv(run_dir / f"{split}_generation_results.csv", gen_rows)
        write_json(run_dir / f"{split}_generation_metrics.json", generation_metrics(gen_rows))

    summary = {
        "config": config,
        "adapter_summary": adapter_artifact.get("adapter_summary"),
        "metrics": classification_metrics(all_results),
        "artifacts": {
            "run_dir": str(run_dir),
            "best_lora": str(run_dir / "best_lora"),
            "config": str(run_dir / "config.json"),
            "per_condition_results": str(run_dir / "per_condition_results.csv"),
            "test_generation_results": str(run_dir / "test_generation_results.csv"),
        },
    }
    write_csv(run_dir / "per_condition_results.csv", all_results)
    write_json(run_dir / "summary.json", summary)
    print(f"Wrote {run_dir / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="control_response_3000_e2")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-contexts", type=int, default=50)
    parser.add_argument("--test-contexts", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--control-loss-weight", type=float, default=4.0)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--position", choices=["all_tokens", "last_token"], default="all_tokens")
    parser.add_argument("--eval-conditions", default="no_time,zero_vector,correct_time_adapter,shuffled_time_adapter,random_norm_matched,non_time_numeric")
    parser.add_argument("--generation-eval-limit", type=int, default=150)
    parser.add_argument("--gen-max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn", default="eager")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-log-every", type=int, default=50)
    parser.add_argument("--gen-log-every", type=int, default=25)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
