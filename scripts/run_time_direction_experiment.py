import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


TIMES = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

FRAGMENTS = [
    "today at school I",
    "I wanted to tell you that",
    "there was something I could not",
    "when I got home I felt",
    "my friend said something and",
    "I am not sure how to",
    "the thing that happened yesterday was",
    "I tried to explain but",
    "before the meeting I was",
    "the doctor told me that",
    "I have been thinking about",
    "after class I saw",
    "I almost said it but",
    "the problem is that I",
    "I need a minute because",
    "something awkward happened when",
    "I was going to ask",
    "the message from them made",
    "I am worried that",
    "the story is a little",
    "I do not know if",
    "what I mean is",
    "I guess I feel",
    "there is one more thing",
]


LABELS = {
    "WAIT": "keep listening because the pause is still short",
    "BACKCHANNEL": "give a brief acknowledgement for a moderate pause",
    "SUPPORT": "offer a supportive sentence for a long pause",
}


@dataclass
class Example:
    fragment_id: int
    fragment: str
    seconds: float
    prompt: str


class TimeAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_size),
        )

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(seconds).view(-1, 1)
        return self.net(x)


def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def build_prompt(fragment: str, seconds: float | None) -> str:
    silence = "" if seconds is None else f"{seconds:.1f} seconds"
    return (
        "Task: choose the conversational timing label.\n"
        "WAIT means keep listening because the pause is still short.\n"
        "BACKCHANNEL means give a brief acknowledgement for a moderate pause.\n"
        "SUPPORT means offer a supportive sentence for a long pause.\n"
        "Examples:\n"
        "User fragment: \"I wanted to tell you that\"\n"
        "Silence: 0.2 seconds\n"
        "Label: WAIT\n"
        "User fragment: \"I wanted to tell you that\"\n"
        "Silence: 1.0 seconds\n"
        "Label: BACKCHANNEL\n"
        "User fragment: \"I wanted to tell you that\"\n"
        "Silence: 4.0 seconds\n"
        "Label: SUPPORT\n"
        "Now label this case.\n"
        f"User fragment: \"{fragment}\"\n"
        f"Silence: {silence}\n"
        "Label:"
    )


def build_examples() -> list[Example]:
    examples: list[Example] = []
    for fragment_id, fragment in enumerate(FRAGMENTS):
        for seconds in TIMES:
            examples.append(
                Example(
                    fragment_id=fragment_id,
                    fragment=fragment,
                    seconds=seconds,
                    prompt=build_prompt(fragment, seconds),
                )
            )
    return examples


def load_model(model_id: str, cache_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.to(device)
    return tokenizer, model, device


def get_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Unsupported model architecture: {type(model)}")


def extract_hidden_layers(tokenizer, model, prompts: list[str], device: torch.device) -> np.ndarray:
    all_layers = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            layers = []
            for hidden in outputs.hidden_states[1:]:
                layers.append(hidden[0, -1, :].detach().float().cpu().numpy().astype(np.float32))
            all_layers.append(np.stack(layers, axis=0))
    return np.stack(all_layers, axis=0)


def cross_validated_layer_metrics(hidden: np.ndarray, y_seconds: np.ndarray, groups: np.ndarray):
    y_reg = np.log1p(y_seconds)
    y_cls = (y_seconds >= 2.0).astype(int)
    n_layers = hidden.shape[1]
    n_splits = min(4, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)
    metrics = []

    for layer in range(n_layers):
        x = hidden[:, layer, :]
        y_pred = np.zeros_like(y_reg, dtype=np.float64)
        y_cls_pred = np.zeros_like(y_cls, dtype=np.int64)
        for train_idx, test_idx in cv.split(x, y_reg, groups):
            reg = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=10.0, random_state=SEED)),
                ]
            )
            reg.fit(x[train_idx], y_reg[train_idx])
            y_pred[test_idx] = reg.predict(x[test_idx])

            clf = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            C=0.5,
                            max_iter=1000,
                            random_state=SEED,
                            solver="liblinear",
                        ),
                    ),
                ]
            )
            clf.fit(x[train_idx], y_cls[train_idx])
            y_cls_pred[test_idx] = clf.predict(x[test_idx])

        corr = float(np.corrcoef(y_reg, y_pred)[0, 1])
        metrics.append(
            {
                "layer": int(layer),
                "r2_log_seconds": float(r2_score(y_reg, y_pred)),
                "corr_log_seconds": corr,
                "long_pause_accuracy": float(accuracy_score(y_cls, y_cls_pred)),
            }
        )
    return metrics


def fit_direction(hidden_layer: np.ndarray, seconds: np.ndarray):
    y = np.log1p(seconds)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=SEED)),
        ]
    )
    pipe.fit(hidden_layer, y)
    scaler = pipe.named_steps["scale"]
    ridge = pipe.named_steps["ridge"]
    coef = ridge.coef_ / scaler.scale_
    norm = np.linalg.norm(coef)
    if norm == 0:
        raise ValueError("Zero direction vector")
    direction = (coef / norm).astype(np.float32)
    return pipe, direction


def add_vector_hook(block, prompt_len: int, vector: torch.Tensor):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        seq_len = hidden.shape[1]
        if seq_len >= prompt_len:
            idx = prompt_len - 1
        else:
            idx = seq_len - 1

        patched = hidden.clone()
        patched[:, idx, :] = patched[:, idx, :] + vector.to(patched.device, patched.dtype)
        if rest is None:
            return patched
        return (patched,) + rest

    return block.register_forward_hook(hook)


def conditional_label_scores(tokenizer, model, prompt: str, layer: int, vector: np.ndarray):
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    prompt_ids = prompt_inputs["input_ids"][0].to(device)
    prompt_len = int(prompt_ids.shape[0])
    blocks = get_blocks(model)
    hook_vec = torch.tensor(vector, dtype=torch.float32).view(1, 1, -1)
    scores = {}

    for label in LABELS:
        candidate = " " + label
        cand_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].to(device)
        input_ids = torch.cat([prompt_ids, cand_ids], dim=0).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        handle = add_vector_hook(blocks[layer], prompt_len, hook_vec)
        with torch.inference_mode():
            out = model(input_ids=input_ids, labels=labels)
        handle.remove()
        # Mean log probability per generated token.
        scores[label] = float(-out.loss.detach().cpu().item())
    return scores


def first_token_distribution(tokenizer, model, prompt: str, layer: int, vector: np.ndarray):
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    blocks = get_blocks(model)
    hook_vec = torch.tensor(vector, dtype=torch.float32).view(1, 1, -1)
    handle = add_vector_hook(blocks[layer], prompt_len, hook_vec)
    with torch.inference_mode():
        out = model(**inputs)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    label_probs = {}
    for label in LABELS:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        label_probs[label] = float(probs[ids[0]].detach().cpu())
    top = torch.topk(probs, k=10)
    top_tokens = [
        {
            "token": tokenizer.decode([int(tok)]),
            "probability": float(prob.detach().cpu()),
        }
        for prob, tok in zip(top.values, top.indices)
    ]
    return label_probs, top_tokens


def generate_with_vector(tokenizer, model, prompt: str, layer: int, vector: np.ndarray, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    blocks = get_blocks(model)
    hook_vec = torch.tensor(vector, dtype=torch.float32).view(1, 1, -1)
    handle = add_vector_hook(blocks[layer], prompt_len, hook_vec)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    handle.remove()
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


def train_adapter(
    train_seconds: np.ndarray,
    train_delta: np.ndarray,
    val_seconds: np.ndarray,
    val_delta: np.ndarray,
    out_path: Path,
):
    hidden_size = train_delta.shape[1]
    adapter = TimeAdapter(hidden_size)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=1e-4)
    x_train = torch.tensor(train_seconds, dtype=torch.float32)
    y_train = torch.tensor(train_delta, dtype=torch.float32)
    x_val = torch.tensor(val_seconds, dtype=torch.float32)
    y_val = torch.tensor(val_delta, dtype=torch.float32)
    losses = []

    for epoch in range(400):
        adapter.train()
        opt.zero_grad()
        pred = adapter(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 399:
            adapter.eval()
            with torch.no_grad():
                val_pred = adapter(x_val)
                val_loss = torch.mean((val_pred - y_val) ** 2)
                zero_loss = torch.mean(y_val**2)
            losses.append(
                {
                    "epoch": epoch,
                    "train_mse": float(loss.detach().cpu()),
                    "val_mse": float(val_loss.detach().cpu()),
                    "val_zero_baseline_mse": float(zero_loss.detach().cpu()),
                }
            )

    adapter.eval()
    with torch.no_grad():
        val_pred = adapter(x_val).detach().cpu().numpy()
    denom = np.linalg.norm(val_pred, axis=1) * np.linalg.norm(val_delta, axis=1)
    cosine = np.divide(
        np.sum(val_pred * val_delta, axis=1),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    torch.save(adapter.state_dict(), out_path)
    return adapter, losses, {
        "val_mse": losses[-1]["val_mse"],
        "val_zero_baseline_mse": losses[-1]["val_zero_baseline_mse"],
        "val_mean_cosine": float(np.mean(cosine)),
        "val_median_cosine": float(np.median(cosine)),
    }


def plot_layer_metrics(metrics, out_path: Path, title: str):
    layers = [m["layer"] for m in metrics]
    r2 = [m["r2_log_seconds"] for m in metrics]
    acc = [m["long_pause_accuracy"] for m in metrics]
    plt.figure(figsize=(8, 4.5))
    plt.plot(layers, r2, marker="o", label="R2 log(1+seconds)")
    plt.plot(layers, acc, marker="s", label="Long-pause accuracy")
    plt.axhline(0, color="#999999", linewidth=0.8)
    plt.ylim(min(-0.25, min(r2) - 0.05), 1.05)
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_projection(times, projection_by_time, no_time_projection, out_path: Path, title: str):
    means = [projection_by_time[str(t)]["mean"] for t in times]
    stds = [projection_by_time[str(t)]["std"] for t in times]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(times, means, yerr=stds, marker="o", capsize=4)
    plt.axhline(no_time_projection["mean"], color="#AA3333", linestyle="--", label="No-time prompt")
    plt.xlabel("Silence seconds")
    plt.ylabel("Projection on time direction")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_intervention(intervention_summary, out_path: Path, title: str):
    target_times = [row["target_seconds"] for row in intervention_summary]
    label_order = list(LABELS.keys())
    data = np.array([[row["mean_scores"][label] for label in label_order] for row in intervention_summary])
    plt.figure(figsize=(7, 4.5))
    im = plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Mean normalized log probability")
    plt.xticks(range(len(label_order)), label_order)
    plt.yticks(range(len(target_times)), [str(t) for t in target_times])
    plt.xlabel("Candidate label")
    plt.ylabel("Injected target seconds")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_adapter_losses(losses, out_path: Path, title: str):
    epochs = [row["epoch"] for row in losses]
    train = [row["train_mse"] for row in losses]
    val = [row["val_mse"] for row in losses]
    baseline = [row["val_zero_baseline_mse"] for row in losses]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train, label="Train MSE")
    plt.plot(epochs, val, label="Validation MSE")
    plt.plot(epochs, baseline, label="Zero baseline", linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_one_model(model_id: str, args):
    out_dir = Path(args.output_dir)
    model_dir = out_dir / safe_name(model_id)
    fig_dir = Path(args.figure_dir) / safe_name(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary_file = model_dir / "summary.json"
    if args.resume and summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)

    tokenizer, model, device = load_model(model_id, cache_dir)
    examples = build_examples()
    prompts = [ex.prompt for ex in examples]
    seconds = np.array([ex.seconds for ex in examples], dtype=np.float32)
    groups = np.array([ex.fragment_id for ex in examples], dtype=np.int64)

    hidden = extract_hidden_layers(tokenizer, model, prompts, device)
    np.save(model_dir / "hidden_time_prompts.npy", hidden)

    base_prompts = [build_prompt(fragment, None) for fragment in FRAGMENTS]
    base_hidden = extract_hidden_layers(tokenizer, model, base_prompts, device)
    np.save(model_dir / "hidden_no_time_prompts.npy", base_hidden)

    metrics = cross_validated_layer_metrics(hidden, seconds, groups)
    best = max(metrics, key=lambda m: (m["r2_log_seconds"], m["long_pause_accuracy"]))
    best_layer = int(best["layer"])

    hidden_best = hidden[:, best_layer, :]
    ridge_pipe, direction = fit_direction(hidden_best, seconds)
    np.save(model_dir / "time_direction.npy", direction)

    projections = hidden_best @ direction
    base_projections = base_hidden[:, best_layer, :] @ direction
    projection_by_time = {}
    alpha_by_time = {}
    for t in TIMES:
        mask = seconds == t
        vals = projections[mask]
        projection_by_time[str(t)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        alpha_by_time[str(t)] = float(np.mean(vals) - np.mean(base_projections))
    no_time_projection = {
        "mean": float(np.mean(base_projections)),
        "std": float(np.std(base_projections)),
    }

    # Direction-vector intervention, aggregated over held-out-ish fragments.
    probe_fragments = FRAGMENTS[:8]
    target_times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    explicit_time_rows = []
    for t in target_times:
        all_scores = []
        for fragment in probe_fragments:
            prompt = build_prompt(fragment, t)
            scores = conditional_label_scores(tokenizer, model, prompt, best_layer, np.zeros_like(direction))
            all_scores.append(scores)
        mean_scores = {
            label: float(np.mean([row[label] for row in all_scores])) for label in LABELS
        }
        explicit_time_rows.append(
            {
                "target_seconds": t,
                "mean_scores": mean_scores,
                "winner": max(mean_scores, key=mean_scores.get),
            }
        )

    intervention_rows = []
    detailed_interventions = []
    for t in target_times:
        alpha = alpha_by_time[str(t)]
        vector = direction * alpha
        all_scores = []
        for fragment in probe_fragments:
            prompt = build_prompt(fragment, None)
            scores = conditional_label_scores(tokenizer, model, prompt, best_layer, vector)
            all_scores.append(scores)
            detailed_interventions.append(
                {
                    "fragment": fragment,
                    "target_seconds": t,
                    "alpha": alpha,
                    "scores": scores,
                    "winner": max(scores, key=scores.get),
                }
            )
        mean_scores = {
            label: float(np.mean([row[label] for row in all_scores])) for label in LABELS
        }
        intervention_rows.append(
            {
                "target_seconds": t,
                "alpha": float(alpha),
                "mean_scores": mean_scores,
                "winner": max(mean_scores, key=mean_scores.get),
            }
        )

    # First-token logits and short generations for representative cases.
    representative_prompt = build_prompt("today at school I", None)
    representative = []
    for t in [0.0, 1.0, 4.0, 8.0]:
        vector = direction * alpha_by_time[str(t)]
        label_probs, top_tokens = first_token_distribution(
            tokenizer, model, representative_prompt, best_layer, vector
        )
        generated = generate_with_vector(
            tokenizer, model, representative_prompt, best_layer, vector
        )
        representative.append(
            {
                "target_seconds": t,
                "label_first_token_probabilities": label_probs,
                "top_next_tokens": top_tokens,
                "greedy_generation": generated,
            }
        )

    # Adapter training split by fragments.
    delta_rows = []
    delta_seconds = []
    for idx, ex in enumerate(examples):
        base_idx = ex.fragment_id
        delta_rows.append(hidden[idx, best_layer, :] - base_hidden[base_idx, best_layer, :])
        delta_seconds.append(ex.seconds)
    delta = np.stack(delta_rows, axis=0).astype(np.float32)
    delta_seconds = np.array(delta_seconds, dtype=np.float32)
    train_fragments = set(range(0, 18))
    train_mask = np.array([g in train_fragments for g in groups])
    val_mask = ~train_mask
    adapter_path = model_dir / "time_adapter.pt"
    adapter, losses, adapter_metrics = train_adapter(
        delta_seconds[train_mask],
        delta[train_mask],
        delta_seconds[val_mask],
        delta[val_mask],
        adapter_path,
    )

    adapter_intervention_rows = []
    adapter_representative = []
    adapter.eval()
    for t in target_times:
        with torch.no_grad():
            vector = adapter(torch.tensor([t], dtype=torch.float32))[0].detach().cpu().numpy()
        all_scores = []
        for fragment in probe_fragments:
            prompt = build_prompt(fragment, None)
            scores = conditional_label_scores(tokenizer, model, prompt, best_layer, vector)
            all_scores.append(scores)
        mean_scores = {
            label: float(np.mean([row[label] for row in all_scores])) for label in LABELS
        }
        adapter_intervention_rows.append(
            {
                "target_seconds": t,
                "mean_scores": mean_scores,
                "winner": max(mean_scores, key=mean_scores.get),
            }
        )
        if t in [0.0, 1.0, 4.0, 8.0]:
            label_probs, top_tokens = first_token_distribution(
                tokenizer, model, representative_prompt, best_layer, vector
            )
            generated = generate_with_vector(
                tokenizer, model, representative_prompt, best_layer, vector
            )
            adapter_representative.append(
                {
                    "target_seconds": t,
                    "label_first_token_probabilities": label_probs,
                    "top_next_tokens": top_tokens,
                    "greedy_generation": generated,
                }
            )

    explicit_representative = []
    zero_vector = np.zeros_like(direction)
    for t in [0.0, 1.0, 4.0, 8.0]:
        prompt = build_prompt("today at school I", t)
        label_probs, top_tokens = first_token_distribution(tokenizer, model, prompt, best_layer, zero_vector)
        generated = generate_with_vector(tokenizer, model, prompt, best_layer, zero_vector)
        explicit_representative.append(
            {
                "target_seconds": t,
                "label_first_token_probabilities": label_probs,
                "top_next_tokens": top_tokens,
                "greedy_generation": generated,
            }
        )

    plot_layer_metrics(
        metrics,
        fig_dir / "layer_metrics.png",
        f"{model_id}: time information by layer",
    )
    plot_projection(
        TIMES,
        projection_by_time,
        no_time_projection,
        fig_dir / "time_projection.png",
        f"{model_id}: projection on estimated time direction",
    )
    plot_intervention(
        intervention_rows,
        fig_dir / "direction_intervention.png",
        f"{model_id}: direction-vector intervention",
    )
    plot_intervention(
        adapter_intervention_rows,
        fig_dir / "adapter_intervention.png",
        f"{model_id}: adapter intervention",
    )
    plot_adapter_losses(
        losses,
        fig_dir / "adapter_losses.png",
        f"{model_id}: time adapter training",
    )

    summary = {
        "model_id": model_id,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_examples": len(examples),
        "times": TIMES,
        "num_fragments": len(FRAGMENTS),
        "num_layers": int(hidden.shape[1]),
        "hidden_size": int(hidden.shape[2]),
        "best_layer": best_layer,
        "best_layer_metrics": best,
        "all_layer_metrics": metrics,
        "projection_by_time": projection_by_time,
        "no_time_projection": no_time_projection,
        "alpha_by_time": alpha_by_time,
        "explicit_time_positive_control_summary": explicit_time_rows,
        "direction_intervention_summary": intervention_rows,
        "direction_intervention_details": detailed_interventions,
        "representative_direction_logits_and_output": representative,
        "representative_adapter_logits_and_output": adapter_representative,
        "representative_explicit_time_logits_and_output": explicit_representative,
        "adapter_metrics": adapter_metrics,
        "adapter_losses": losses,
        "adapter_intervention_summary": adapter_intervention_rows,
        "artifact_files": {
            "hidden_time_prompts": str(model_dir / "hidden_time_prompts.npy"),
            "hidden_no_time_prompts": str(model_dir / "hidden_no_time_prompts.npy"),
            "time_direction": str(model_dir / "time_direction.npy"),
            "time_adapter": str(adapter_path),
            "figures": str(fig_dir),
        },
    }
    with open(model_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct", "gpt2"])
    parser.add_argument("--output-dir", default="artifacts/time_direction")
    parser.add_argument("--figure-dir", default="output/figures/time_direction")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--summary-path", default="artifacts/time_direction/summary_all.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    summaries = []
    failures = []
    for model_id in args.models:
        print(f"=== Running {model_id} ===", flush=True)
        try:
            summaries.append(run_one_model(model_id, args))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append(
                {
                    "model_id": model_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"FAILED {model_id}: {type(exc).__name__}: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"models": summaries, "failures": failures}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
