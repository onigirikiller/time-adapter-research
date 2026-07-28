from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_omni_utils import process_mm_info
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OMNI_SINGLE_TOKEN_DATA_DIR", str(ROOT / "data/omni3b_delayed_backchannel_v2")))
SOURCE_DATA_DIR = Path(os.environ.get("OMNI_SINGLE_TOKEN_SOURCE_DATA_DIR", str(ROOT / "data/omni3b_sequential_v2")))
HIDDEN_DIR = ROOT / "artifacts/omni3b_sequential_v2/hidden_cache"
ADAPTER_DIR = ROOT / "artifacts/omni3b_generation_hook_v3"
OUT_DIR = ROOT / "artifacts/omni3b_single_token_lora_v1"
CACHE_DIR = ROOT / ".cache/huggingface"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SEED = 20260628

LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

CODEBOOKS = {
    # Existing single-token Qwen added-vocabulary tokens. They are visually distinct
    # from ordinary response text and avoid resizing embeddings.
    "fim": {
        "WAIT": "<|fim_prefix|>",
        "BACKCHANNEL": "<|fim_middle|>",
        "SUPPORT": "<|fim_suffix|>",
    },
    # Ablation codebooks. These are one token but less isolated from natural text.
    "wbs": {"WAIT": "W", "BACKCHANNEL": "B", "SUPPORT": "S"},
    "abc": {"WAIT": "A", "BACKCHANNEL": "B", "SUPPORT": "C"},
    "slash": {"WAIT": "/W", "BACKCHANNEL": "/B", "SUPPORT": "/S"},
    "punct": {"WAIT": "@@", "BACKCHANNEL": "&&", "SUPPORT": "%%"},
}

EVAL_CONDITIONS = [
    "no_time",
    "zero_vector",
    "correct_time_adapter",
    "shuffled_time_adapter",
    "random_norm_matched",
    "non_time_numeric",
    "oracle_explicit_delta",
]

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


omni_v1 = import_module(ROOT / "scripts/run_omni_sequential_time_adapter.py", "omni_v1_single_token")
omni_v2 = import_module(ROOT / "scripts/run_omni3b_v2_experiment.py", "omni_v2_single_token")


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, hidden_size))

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fields = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_rows(split: str) -> list[dict]:
    return read_jsonl(DATA_DIR / f"{split}.jsonl")


def label_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["label"] for row in rows))


def balanced_sample(rows: list[dict], per_class: int, seed: int, allow_oversample: bool = True) -> list[dict]:
    rng = random.Random(seed)
    by_label = {label: [row for row in rows if row["label"] == label] for label in LABELS}
    selected = []
    for label in LABELS:
        pool = by_label[label]
        if not pool:
            raise ValueError(f"No rows for label {label}")
        if len(pool) >= per_class:
            selected.extend(rng.sample(pool, per_class))
        elif allow_oversample:
            selected.extend(pool)
            selected.extend(rng.choice(pool) for _ in range(per_class - len(pool)))
        else:
            raise ValueError(f"Requested {per_class} rows for {label}, only {len(pool)} available")
    rng.shuffle(selected)
    return [dict(row, train_sample_index=i) for i, row in enumerate(selected)]


def limited_contexts(rows: list[dict], contexts: int) -> list[dict]:
    if contexts <= 0:
        return rows
    by_profile: dict[str, list[str]] = defaultdict(list)
    seen_by_profile: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cid = row["context_id"]
        profile = row["profile"]
        if cid not in seen_by_profile[profile]:
            seen_by_profile[profile].add(cid)
            by_profile[profile].append(cid)
    keep = []
    profiles = sorted(by_profile)
    offset = 0
    while len(keep) < contexts:
        advanced = False
        for profile in profiles:
            if offset < len(by_profile[profile]):
                keep.append(by_profile[profile][offset])
                advanced = True
                if len(keep) >= contexts:
                    break
        if not advanced:
            break
        offset += 1
    keep_set = set(keep)
    return [row for row in rows if row["context_id"] in keep_set]


def feature_matrix(rows: list[dict]) -> np.ndarray:
    vals = []
    for row in rows:
        f = row["features"]
        vals.append(
            [
                np.log1p(float(f["silence_elapsed"])),
                float(f["delta_t"]),
                np.log1p(float(f["utterance_elapsed"])),
                1.0 if f["is_user_speaking"] else 0.0,
                1.0 if f["asr_changed"] else 0.0,
            ]
        )
    return np.asarray(vals, dtype=np.float32)


def load_time_adapter(layer: int):
    path = ADAPTER_DIR / f"adapter_proxy_stage-extra_layer-{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Time Adapter artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    hidden_size = int(next(iter(artifact["adapter_state"].values())).shape[0])
    # The first state tensor is net.0.weight [64, 5], so infer hidden size from final layer.
    for key, value in artifact["adapter_state"].items():
        if key.endswith("2.weight"):
            hidden_size = int(value.shape[0])
            break
    adapter = FeatureAdapter(5, hidden_size)
    adapter.load_state_dict(artifact["adapter_state"])
    adapter.eval()
    return adapter, artifact


def adapter_predict(adapter: nn.Module, rows: list[dict]) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(feature_matrix(rows), dtype=torch.float32)
        return adapter(x).detach().cpu().numpy().astype(np.float32)


def load_hidden_index(split: str) -> dict[str, int]:
    rows = read_jsonl(SOURCE_DATA_DIR / f"{split}.jsonl")
    return {row["id"]: i for i, row in enumerate(rows)}


def oracle_delta_for_rows(rows: list[dict], split: str, layer: int) -> np.ndarray:
    idx = load_hidden_index(split)
    ids = [idx[row["id"]] for row in rows]
    no_time = np.load(HIDDEN_DIR / f"no_time_{split}.npy", mmap_mode="r")[ids, layer, :]
    explicit = np.load(HIDDEN_DIR / f"explicit_{split}.npy", mmap_mode="r")[ids, layer, :]
    return np.asarray(explicit - no_time, dtype=np.float32)


def build_condition_vectors(rows: list[dict], split: str, layer: int, seed: int) -> dict[str, np.ndarray]:
    adapter, artifact = load_time_adapter(layer)
    correct = adapter_predict(adapter, rows)
    zero = np.zeros_like(correct, dtype=np.float32)
    rng = np.random.default_rng(seed + layer)
    order = np.arange(len(rows))
    rng.shuffle(order)
    shuffled = correct[order].copy()
    random_dir = rng.normal(size=correct.shape).astype(np.float32)
    random_norm = np.linalg.norm(random_dir, axis=1, keepdims=True)
    target_norm = np.linalg.norm(correct, axis=1, keepdims=True)
    random_vec = random_dir / np.maximum(random_norm, 1e-12) * target_norm
    direction = np.asarray(artifact["non_time_direction_score"], dtype=np.float32).reshape(1, -1)
    non_time = direction * target_norm
    oracle = oracle_delta_for_rows(rows, split, layer)
    return {
        "no_time": zero,
        "zero_vector": zero,
        "correct_time_adapter": correct,
        "shuffled_time_adapter": shuffled,
        "random_norm_matched": random_vec.astype(np.float32),
        "non_time_numeric": non_time.astype(np.float32),
        "oracle_explicit_delta": oracle.astype(np.float32),
    }


class InjectionHook:
    def __init__(self, vector: np.ndarray, alpha: float, position: str, enabled: bool = True):
        self.vector = vector
        self.alpha = float(alpha)
        self.position = position
        self.enabled = enabled
        self.calls = 0
        self.stats: list[dict] = []

    def __call__(self, module, inputs, output):
        self.calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        vec = torch.as_tensor(self.vector, device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        vec = vec * self.alpha
        target = hidden[:, -1:, :]
        before = target.detach().float()
        vecf = vec.detach().float()
        denom = torch.linalg.norm(before.reshape(1, -1)) * torch.linalg.norm(vecf.reshape(1, -1))
        cosine = torch.sum(before.reshape(1, -1) * vecf.reshape(1, -1)) / torch.clamp(denom, min=1e-12)
        self.stats.append(
            {
                "call": self.calls,
                "seq_len": int(hidden.shape[1]),
                "hidden_norm": float(torch.linalg.norm(before).detach().cpu()),
                "injected_norm": float(torch.linalg.norm(vecf).detach().cpu()),
                "hidden_injected_cosine": float(cosine.detach().cpu()),
            }
        )
        if not self.enabled:
            return output
        if self.position == "all_tokens":
            modified = hidden + vec
        else:
            modified = hidden.clone()
            modified[:, -1:, :] = modified[:, -1:, :] + vec
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def codebook_instruction(codebook: dict[str, str]) -> str:
    return (
        "Choose one listener timing control code. Output exactly one code token and no other text.\n"
        f"{codebook['WAIT']} = WAIT: stay silent and keep listening.\n"
        f"{codebook['BACKCHANNEL']} = BACKCHANNEL: give a very short acknowledgement only.\n"
        f"{codebook['SUPPORT']} = SUPPORT: take the turn with a brief helpful/supportive response.\n"
    )


def resolve_audio_path(row: dict, audio_timing_mode: str) -> Path:
    if audio_timing_mode == "row_audio":
        return Path(row["audio_path"]).resolve()
    if audio_timing_mode == "base_0s_audio":
        return (SOURCE_DATA_DIR / "audio" / f"{row['context_id']}_silence_0s.wav").resolve()
    raise ValueError(f"Unknown audio_timing_mode: {audio_timing_mode}")


def build_conversation(row: dict, prompt_mode: str, codebook: dict[str, str], explicit_seconds: bool = False, audio_timing_mode: str = "base_0s_audio"):
    content = [{"type": "audio", "audio": str(resolve_audio_path(row, audio_timing_mode))}]
    instruction = codebook_instruction(codebook)
    if prompt_mode == "audio_only":
        user_text = (
            "The audio contains a partial spoken user utterance and possible following silence.\n"
            "Use the audio content and the hidden external timer signal.\n"
            + instruction
            + "Control code:"
        )
    elif prompt_mode == "audio_text":
        user_text = (
            "The audio contains a partial spoken user utterance and possible following silence.\n"
            f"ASR fragment: \"{row['fragment']}\"\n"
            "Use the ASR fragment, audio context, and hidden external timer signal.\n"
            + instruction
            + "Control code:"
        )
    elif prompt_mode == "text_only":
        content = []
        user_text = (
            f"ASR fragment: \"{row['fragment']}\"\n"
            "Use the ASR fragment and hidden external timer signal.\n"
            + instruction
            + "Control code:"
        )
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
    if explicit_seconds:
        f = row["features"]
        user_text = (
            user_text
            + "\nExternal timing features for this explicit baseline:\n"
            + f"silence_elapsed={f['silence_elapsed']} seconds\n"
            + f"delta_t={f['delta_t']} seconds\n"
            + f"utterance_elapsed={f['utterance_elapsed']} seconds\n"
            + f"is_user_speaking={f['is_user_speaking']}\n"
            + f"asr_changed={f['asr_changed']}\n"
            + "Control code:"
        )
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def prepare_inputs(
    processor,
    row: dict,
    prompt_mode: str,
    codebook: dict[str, str],
    label: str | None = None,
    explicit_seconds: bool = False,
    audio_timing_mode: str = "base_0s_audio",
):
    conv = build_conversation(row, prompt_mode, codebook, explicit_seconds=explicit_seconds, audio_timing_mode=audio_timing_mode)
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text if label is None else prompt_text + codebook[label]
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    kwargs = {
        "text": full_text,
        "images": images,
        "videos": videos,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": False,
    }
    if audios is not None:
        kwargs["audio"] = audios
    inputs = processor(**kwargs)
    if label is None:
        return inputs, int(inputs["input_ids"].shape[1]), prompt_text
    prompt_kwargs = dict(kwargs)
    prompt_kwargs["text"] = prompt_text
    prompt_inputs = processor(**prompt_kwargs)
    return inputs, int(prompt_inputs["input_ids"].shape[1]), prompt_text


def move_inputs(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        else:
            moved[key] = value
    return moved


def label_token_ids(tokenizer, codebook: dict[str, str]) -> dict[str, int]:
    ids = {}
    for label, surface in codebook.items():
        toks = tokenizer(surface, add_special_tokens=False).input_ids
        if len(toks) != 1:
            raise ValueError(f"{label} surface {surface!r} is not one token: {toks}")
        ids[label] = int(toks[0])
    return ids


def load_model(dtype_name: str, attn: str, use_cache: bool, gradient_checkpointing: bool):
    dtype = torch.bfloat16 if dtype_name == "bf16" and torch.cuda.is_available() else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn,
    )
    model.disable_talker()
    model.config.use_cache = use_cache
    if hasattr(model.thinker, "config"):
        model.thinker.config.use_cache = use_cache
    if gradient_checkpointing and hasattr(model.thinker, "gradient_checkpointing_enable"):
        model.thinker.gradient_checkpointing_enable()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    return processor, model, model.device, dtype


def apply_lora(model, r: int, alpha: int, dropout: float, target_modules: str):
    targets = [item.strip() for item in target_modules.split(",") if item.strip()]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model.thinker = get_peft_model(model.thinker, config)
    return model


def attach_existing_lora(model, adapter_path: Path):
    model.thinker = PeftModel.from_pretrained(model.thinker, adapter_path, is_trainable=False)
    return model


def trainable_parameter_summary(model) -> dict:
    trainable = 0
    total = 0
    for p in model.thinker.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total, "ratio": trainable / max(total, 1)}


def hook_module(model, layer: int):
    thinker = model.thinker
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
    ]
    for path in candidates:
        obj = thinker
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj[layer]
    names = [name for name, _ in thinker.named_modules() if name.endswith(f"model.layers.{layer}")]
    if names:
        module_name = names[0]
        modules = dict(thinker.named_modules())
        return modules[module_name]
    raise AttributeError(f"Could not locate Thinker decoder layer {layer} after PEFT wrapping")


def vector_for_training(row: dict, train_condition: str, adapter: nn.Module | None, split: str, layer: int) -> np.ndarray:
    if train_condition in {"no_time", "zero_vector"}:
        if adapter is None:
            raise ValueError("Adapter required to infer hidden size for zero vector")
        hidden = adapter.net[-1].out_features
        return np.zeros((hidden,), dtype=np.float32)
    if train_condition == "correct_time_adapter":
        return adapter_predict(adapter, [row])[0]
    if train_condition == "oracle_explicit_delta":
        return oracle_delta_for_rows([row], split, layer)[0]
    raise ValueError(f"Unknown train_condition {train_condition}")


def train_one_epoch(
    model,
    processor,
    rows: list[dict],
    adapter: nn.Module,
    optimizer,
    device,
    dtype,
    args,
    codebook: dict[str, str],
    epoch: int,
    log_path: Path,
):
    model.thinker.train()
    rng = random.Random(args.seed + epoch)
    order = list(range(len(rows)))
    rng.shuffle(order)
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()
    for step_i, row_i in enumerate(order):
        row = rows[row_i]
        inputs, prompt_len, _ = prepare_inputs(
            processor,
            row,
            args.prompt_mode,
            codebook,
            label=row["label"],
            audio_timing_mode=args.audio_timing_mode,
        )
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels[:, prompt_len + 1 :] = -100
        moved = move_inputs(inputs, device, dtype)
        labels = labels.to(device)
        vector = vector_for_training(row, args.train_condition, adapter, "train", args.layer)
        enabled = args.train_condition not in {"no_time"}
        hook = InjectionHook(vector, alpha=args.alpha, position=args.position, enabled=enabled)
        handle = hook_module(model, args.layer).register_forward_hook(hook)
        try:
            outputs = model.thinker(**moved, labels=labels, use_audio_in_video=False)
            loss = outputs.loss / args.grad_accum
            loss.backward()
        finally:
            handle.remove()
        total_loss += float(loss.detach().cpu()) * args.grad_accum
        steps += 1
        if steps % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.thinker.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available() and steps % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        if steps == 1 or steps % args.log_every == 0 or steps == len(rows):
            elapsed = time.time() - start
            msg = {
                "epoch": epoch,
                "step": steps,
                "rows": len(rows),
                "avg_loss": total_loss / max(steps, 1),
                "elapsed_sec": elapsed,
                "examples_per_sec": steps / max(elapsed, 1e-6),
                "hook_calls": hook.calls,
                "hook_stats": hook.stats[:1],
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            print(f"epoch={epoch} step={steps}/{len(rows)} loss={msg['avg_loss']:.4f}", flush=True)
    if steps % args.grad_accum:
        torch.nn.utils.clip_grad_norm_([p for p in model.thinker.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"epoch": epoch, "train_loss": total_loss / max(steps, 1), "train_steps": steps}


def softmax3(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    y = np.exp(x)
    return y / np.sum(y)


def evaluate(
    model,
    processor,
    rows: list[dict],
    split: str,
    vectors: dict[str, np.ndarray],
    device,
    dtype,
    args,
    codebook: dict[str, str],
    token_ids: dict[str, int],
    conditions: list[str],
    limit: int = 0,
):
    model.thinker.eval()
    selected = rows[:limit] if limit else rows
    results = []
    with torch.inference_mode():
        for i, row in enumerate(selected):
            prompt_inputs, prompt_len, prompt_text = prepare_inputs(
                processor,
                row,
                args.prompt_mode,
                codebook,
                label=None,
                explicit_seconds=False,
                audio_timing_mode=args.audio_timing_mode,
            )
            moved = move_inputs(prompt_inputs, device, dtype)
            for condition in conditions:
                vector = vectors[condition][i]
                enabled = condition != "no_time"
                hook = InjectionHook(vector, alpha=args.alpha, position=args.position, enabled=enabled)
                handle = hook_module(model, args.layer).register_forward_hook(hook)
                try:
                    t0 = time.perf_counter()
                    outputs = model.thinker(**moved, use_audio_in_video=False)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                finally:
                    handle.remove()
                logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
                label_logits = np.array([logits[token_ids[label]] for label in LABELS], dtype=np.float64)
                probs = softmax3(label_logits)
                pred = LABELS[int(np.argmax(probs))]
                f = row["features"]
                stat = hook.stats[0] if hook.stats else {}
                results.append(
                    {
                        "id": row["id"],
                        "context_id": row["context_id"],
                        "split": split,
                        "profile": row["profile"],
                        "fragment": row["fragment"],
                        "silence_seconds": row["silence_seconds"],
                        "gold_label": row["label"],
                        "condition": condition,
                        "pred_label": pred,
                        "correct": int(pred == row["label"]),
                        "wait_logit": float(label_logits[0]),
                        "backchannel_logit": float(label_logits[1]),
                        "support_logit": float(label_logits[2]),
                        "wait_prob": float(probs[0]),
                        "backchannel_prob": float(probs[1]),
                        "support_prob": float(probs[2]),
                        "label_token_wait": token_ids["WAIT"],
                        "label_token_backchannel": token_ids["BACKCHANNEL"],
                        "label_token_support": token_ids["SUPPORT"],
                        "hook_calls": hook.calls,
                        "hidden_norm": stat.get("hidden_norm", 0.0),
                        "injected_norm": stat.get("injected_norm", 0.0),
                        "hidden_injected_cosine": stat.get("hidden_injected_cosine", 0.0),
                        "seq_len": stat.get("seq_len", 0),
                        "latency_ms": latency_ms,
                        "layer": args.layer,
                        "alpha": args.alpha,
                        "position": args.position,
                        "silence_elapsed": f["silence_elapsed"],
                        "delta_t": f["delta_t"],
                        "utterance_elapsed": f["utterance_elapsed"],
                        "is_user_speaking": f["is_user_speaking"],
                        "asr_changed": f["asr_changed"],
                    }
                )
            if (i + 1) % args.eval_log_every == 0 or i + 1 == len(selected):
                print(f"eval {split}: {i + 1}/{len(selected)} rows", flush=True)
    return results


def sequence_metrics(rows: list[dict]) -> dict:
    order = {"WAIT": 0, "BACKCHANNEL": 1, "SUPPORT": 2}
    by_context = defaultdict(list)
    for row in rows:
        by_context[row["context_id"]].append(row)
    exact = 0
    step_total = 0
    step_correct = 0
    premature = 0
    delayed = 0
    regression = 0
    profile_acc = defaultdict(lambda: {"contexts": 0, "exact": 0, "steps": 0, "correct": 0})
    transitions = []
    for cid, items in by_context.items():
        items = sorted(items, key=lambda r: float(r["silence_seconds"]))
        gold = [r["gold_label"] for r in items]
        pred = [r["pred_label"] for r in items]
        prof = items[0]["profile"]
        is_exact = int(gold == pred)
        exact += is_exact
        profile_acc[prof]["contexts"] += 1
        profile_acc[prof]["exact"] += is_exact
        transitions.append(
            {
                "context_id": cid,
                "profile": prof,
                "seconds": [float(r["silence_seconds"]) for r in items],
                "gold_sequence": gold,
                "pred_sequence": pred,
            }
        )
        for j, (g, p) in enumerate(zip(gold, pred)):
            step_total += 1
            ok = int(g == p)
            step_correct += ok
            profile_acc[prof]["steps"] += 1
            profile_acc[prof]["correct"] += ok
            if order[p] > order[g]:
                premature += 1
            if order[p] < order[g]:
                delayed += 1
            if j and order[p] < order[pred[j - 1]]:
                regression += 1
    denom_transitions = max(step_total - len(by_context), 1)
    return {
        "exact_sequence_accuracy": exact / max(len(by_context), 1),
        "step_accuracy": step_correct / max(step_total, 1),
        "premature_escalation_rate": premature / max(step_total, 1),
        "delayed_support_rate": delayed / max(step_total, 1),
        "regression_rate": regression / denom_transitions,
        "profile": {
            k: {
                "contexts": v["contexts"],
                "exact_sequence_accuracy": v["exact"] / max(v["contexts"], 1),
                "step_accuracy": v["correct"] / max(v["steps"], 1),
            }
            for k, v in sorted(profile_acc.items())
        },
        "transitions": transitions,
    }


def classification_metrics(results: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)
    out = {}
    for condition, items in sorted(by_condition.items()):
        y = [LABEL_TO_INDEX[row["gold_label"]] for row in items]
        p = [LABEL_TO_INDEX[row["pred_label"]] for row in items]
        prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        out[condition] = {
            "rows": len(items),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
            "per_class": {
                LABELS[i]: {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "support": int(sup[i]),
                }
                for i in range(3)
            },
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "pred_counts": dict(Counter(row["pred_label"] for row in items)),
            "mean_probs": {
                label: float(np.mean([float(row[f"{label.lower()}_prob"]) for row in items]))
                for label in LABELS
            },
            "latency_ms": {
                "mean": float(np.mean([float(row["latency_ms"]) for row in items])),
                "p50": float(np.percentile([float(row["latency_ms"]) for row in items], 50)),
                "p90": float(np.percentile([float(row["latency_ms"]) for row in items], 90)),
                "p95": float(np.percentile([float(row["latency_ms"]) for row in items], 95)),
                "p99": float(np.percentile([float(row["latency_ms"]) for row in items], 99)),
            },
            "sequence": sequence_metrics(items),
        }
    return out


def failure_rows(results: list[dict], limit: int = 100) -> list[dict]:
    failures = []
    for row in results:
        if row["gold_label"] != row["pred_label"]:
            conf = max(float(row["wait_prob"]), float(row["backchannel_prob"]), float(row["support_prob"]))
            failures.append({**row, "confidence": conf})
    failures.sort(key=lambda r: r["confidence"], reverse=True)
    return failures[:limit]


def save_checkpoint(model, run_dir: Path, name: str):
    path = run_dir / name
    path.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(path)
    return path


def tokenizer_report(tokenizer) -> dict:
    surfaces = {
        "plain_WAIT_BACKCHANNEL_SUPPORT": ["WAIT", "BACKCHANNEL", "SUPPORT"],
        "WBS": ["W", "B", "S"],
        "ABC": ["A", "B", "C"],
        "FIM": ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"],
        "angle_WBS": ["<W>", "<B>", "<S>"],
        "slash_WBS": ["/W", "/B", "/S"],
        "punct": ["@@", "&&", "%%"],
        "jp": ["待つ", "相槌", "支援"],
    }
    out = {}
    for name, labels in surfaces.items():
        out[name] = []
        for surface in labels:
            ids = tokenizer(surface, add_special_tokens=False).input_ids
            out[name].append({"surface": surface, "num_tokens": len(ids), "ids": ids})
    return out


def build_config(args, codebook: dict, token_ids: dict, train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]):
    source_manifest = read_json(SOURCE_DATA_DIR / "manifest.json")
    delayed_manifest = read_json(DATA_DIR / "manifest.json")
    return {
        "run_name": args.run_name,
        "model_id": MODEL_ID,
        "dataset": str(DATA_DIR),
        "source_audio_dataset": str(SOURCE_DATA_DIR),
        "source_audio_tts": source_manifest.get("tts"),
        "source_waveform_silence": source_manifest.get("waveform_silence"),
        "delayed_label_manifest": delayed_manifest,
        "codebook_name": args.codebook,
        "codebook": codebook,
        "token_ids": token_ids,
        "prompt_mode": args.prompt_mode,
        "audio_timing_mode": args.audio_timing_mode,
        "prompt_has_seconds": False,
        "train_condition": args.train_condition,
        "eval_conditions": args.eval_conditions.split(","),
        "layer": args.layer,
        "alpha": args.alpha,
        "position": args.position,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
        "train": {"rows": len(train_rows), "label_counts": label_counts(train_rows)},
        "validation": {"rows": len(val_rows), "label_counts": label_counts(val_rows)},
        "test": {"rows": len(test_rows), "label_counts": label_counts(test_rows)},
        "note": (
            "Existing source audio is Qwen3TTS 0.6B according to manifest. "
            "Discarded SAPI audio directory is not used. Qwen3TTS 1.7B augmentation is reserved for follow-up if direct LoRA shows benefit."
        ),
    }


def run(args):
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"

    processor, model, device, dtype = load_model(args.dtype, args.attn, use_cache=False, gradient_checkpointing=args.gradient_checkpointing)
    codebook = CODEBOOKS[args.codebook]
    token_ids = label_token_ids(processor.tokenizer, codebook)
    write_json(run_dir / "tokenizer_report.json", tokenizer_report(processor.tokenizer))
    adapter, adapter_artifact = load_time_adapter(args.layer)

    train_all = load_rows("train")
    val_all = load_rows("validation")
    test_all = load_rows("test")
    train_rows = balanced_sample(train_all, args.train_per_class, args.seed, allow_oversample=True)
    val_rows = limited_contexts(val_all, args.val_contexts)
    test_rows = limited_contexts(test_all, args.test_contexts)
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows:
        val_rows = val_rows[: args.max_val_rows]
    if args.max_test_rows:
        test_rows = test_rows[: args.max_test_rows]

    config = build_config(args, codebook, token_ids, train_rows, val_rows, test_rows)
    write_json(run_dir / "config.json", config)
    print(json.dumps({"config": config["train"], "token_ids": token_ids}, ensure_ascii=False), flush=True)

    if args.mode in {"train", "all"}:
        model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
        param_summary = trainable_parameter_summary(model)
        write_json(run_dir / "trainable_parameters.json", param_summary)
        print(f"Trainable parameters: {param_summary}", flush=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.thinker.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        history = []
        best_score = -1.0
        best_path = None
        for epoch in range(args.epochs):
            hist = train_one_epoch(model, processor, train_rows, adapter, optimizer, device, dtype, args, codebook, epoch, log_path)
            history.append(hist)
            val_vectors = build_condition_vectors(val_rows, "validation", args.layer, args.seed + epoch)
            val_results = evaluate(
                model,
                processor,
                val_rows,
                "validation",
                val_vectors,
                device,
                dtype,
                args,
                codebook,
                token_ids,
                [args.selection_condition],
                limit=args.selection_eval_limit,
            )
            val_metrics = classification_metrics(val_results)
            score = val_metrics[args.selection_condition]["macro_f1"]
            hist["validation_selection_macro_f1"] = score
            hist["validation_selection_accuracy"] = val_metrics[args.selection_condition]["accuracy"]
            write_json(run_dir / "training_history.json", history)
            write_json(run_dir / f"epoch_{epoch:02d}_validation_metrics.json", val_metrics)
            if score > best_score:
                best_score = score
                best_path = save_checkpoint(model, run_dir, "best_lora")
                write_json(run_dir / "best_checkpoint.json", {"epoch": epoch, "score": score, "path": str(best_path)})
            if args.save_each_epoch:
                save_checkpoint(model, run_dir, f"epoch_{epoch:02d}_lora")
        # Keep the in-memory adapter for evaluation. The best checkpoint is saved
        # separately; reloading it into the same PEFT wrapper creates nested adapter
        # names in current PEFT versions.
    elif args.mode == "eval":
        if not args.adapter_path:
            raise ValueError("--adapter-path is required for eval mode")
        model = attach_existing_lora(model, Path(args.adapter_path))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_conditions = [c for c in args.eval_conditions.split(",") if c]
    all_results = []
    split_rows = {"validation": val_rows, "test": test_rows}
    for split, rows in split_rows.items():
        vectors = build_condition_vectors(rows, split, args.layer, args.seed)
        results = evaluate(model, processor, rows, split, vectors, device, dtype, args, codebook, token_ids, eval_conditions)
        all_results.extend(results)
        write_csv(run_dir / f"{split}_per_condition_results.csv", results)
        write_json(run_dir / f"{split}_metrics.json", classification_metrics(results))

    metrics = classification_metrics(all_results)
    write_csv(run_dir / "per_condition_results.csv", all_results)
    write_csv(run_dir / "per_timepoint_logprobs.csv", all_results)
    write_csv(run_dir / "failure_cases.csv", failure_rows(all_results))
    summary = {
        "config": config,
        "time_adapter_artifact": str(ADAPTER_DIR / f"adapter_proxy_stage-extra_layer-{args.layer}.pt"),
        "adapter_summary": adapter_artifact.get("adapter_summary"),
        "metrics": metrics,
        "artifacts": {
            "config": str(run_dir / "config.json"),
            "tokenizer_report": str(run_dir / "tokenizer_report.json"),
            "train_log": str(log_path),
            "per_condition_results": str(run_dir / "per_condition_results.csv"),
            "per_timepoint_logprobs": str(run_dir / "per_timepoint_logprobs.csv"),
            "failure_cases": str(run_dir / "failure_cases.csv"),
            "best_lora": str(run_dir / "best_lora"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Wrote {run_dir / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="smoke_fim_l3_a4")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--codebook", choices=sorted(CODEBOOKS), default="fim")
    parser.add_argument("--prompt-mode", choices=["audio_text", "audio_only", "text_only"], default="audio_text")
    parser.add_argument("--audio-timing-mode", choices=["base_0s_audio", "row_audio"], default="base_0s_audio")
    parser.add_argument("--train-condition", choices=["no_time", "zero_vector", "correct_time_adapter", "oracle_explicit_delta"], default="correct_time_adapter")
    parser.add_argument("--selection-condition", default="correct_time_adapter")
    parser.add_argument("--eval-conditions", default="no_time,zero_vector,correct_time_adapter,shuffled_time_adapter,random_norm_matched,non_time_numeric,oracle_explicit_delta")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--position", choices=["last_token", "all_tokens"], default="all_tokens")
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--val-contexts", type=int, default=2)
    parser.add_argument("--test-contexts", type=int, default=2)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--selection-eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn", default="eager")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-log-every", type=int, default=10)
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
