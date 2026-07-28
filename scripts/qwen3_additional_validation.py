import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
TIMES = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
CONTROL_VALUES = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]

FRAGMENTS = [
    "今日学校で……",
    "えっと、今日ちょっと……",
    "I wanted to tell you that",
    "Something awkward happened when",
    "ごめん、少し話しづらくて……",
    "つまり結論としては……",
    "今日の会議で言いたかったのは……",
    "I am not sure how to say this",
    "さっきの件なんだけど……",
    "there is one more thing",
    "ちょっと考えを整理すると……",
    "I guess what I mean is",
]


TIME_VARIANTS = {
    "seconds_bracket": {
        "kind": "time",
        "text": lambda s: f"[{s:g}s]",
        "pair": (0.5, 5.0, "[0.5s]", "[5s]"),
    },
    "milliseconds_bracket": {
        "kind": "time",
        "text": lambda s: f"[{int(round(s * 1000))}ms]",
        "pair": (0.5, 5.0, "[500ms]", "[5000ms]"),
    },
    "jp_silence": {
        "kind": "time",
        "text": lambda s: {
            0.2: "ごく短い沈黙",
            0.5: "短い沈黙",
            1.0: "少し沈黙",
            2.0: "やや長い沈黙",
            4.0: "長い沈黙",
            8.0: "とても長い沈黙",
        }[s],
        "pair": (0.5, 4.0, "短い沈黙", "長い沈黙"),
    },
    "jp_moment_while": {
        "kind": "time",
        "text": lambda s: {
            0.2: "ほんの一瞬黙って",
            0.5: "一瞬黙って",
            1.0: "少し黙って",
            2.0: "しばらく黙って",
            4.0: "長めに黙って",
            8.0: "かなり長く黙って",
        }[s],
        "pair": (0.5, 2.0, "一瞬黙って", "しばらく黙って"),
    },
    "jp_pause_phrase": {
        "kind": "time",
        "text": lambda s: {
            0.2: "ごく短く間を置いて",
            0.5: "少し間を置いて",
            1.0: "短めに間を置いて",
            2.0: "しばらく間を置いて",
            4.0: "長く間を置いて",
            8.0: "かなり長く間を置いて",
        }[s],
        "pair": (0.5, 4.0, "少し間を置いて", "長く間を置いて"),
    },
}

NON_TIME_VARIANTS = {
    "kg": {"unit": "kg", "text": lambda v: f"[{v:g}kg]"},
    "m": {"unit": "m", "text": lambda v: f"[{v:g}m]"},
    "score": {"unit": "点", "text": lambda v: f"[{v:g}点]"},
    "yen": {"unit": "円", "text": lambda v: f"[{v:g}円]"},
}


@dataclass
class HiddenExample:
    variant: str
    fragment_id: int
    fragment: str
    value: float
    cue: str
    prompt: str


class TimeAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 96), nn.Tanh(), nn.Linear(96, hidden_size))

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log1p(seconds).view(-1, 1))


def build_prompt(fragment: str, cue: str | None, cue_label: str = "Timing cue") -> str:
    cue_text = "not provided" if cue is None else cue
    return (
        "You are a timing classifier for a streaming dialogue system.\n"
        "Use both the user's unfinished utterance and the timing cue.\n"
        "Labels:\n"
        "WAIT = the user likely still wants to hold the floor, so keep listening.\n"
        "BACKCHANNEL = give a short acknowledgement such as 'うん' or 'I see'.\n"
        "SUPPORT = the assistant should actively respond or offer gentle support.\n"
        f"User fragment: \"{fragment}\"\n"
        f"{cue_label}: {cue_text}\n"
        "Answer with exactly one label.\n"
        "Label:"
    )


def build_control_prompt(fragment: str, cue: str) -> str:
    return build_prompt(fragment, cue, cue_label="Unrelated numeric note")


def load_model(cache_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(cache_dir),
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def get_blocks(model):
    return model.model.layers


def extract_hidden(tokenizer, model, device, prompts: list[str]) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            layers = [
                h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
                for h in outputs.hidden_states[1:]
            ]
            rows.append(np.stack(layers, axis=0))
    return np.stack(rows, axis=0)


def cv_metrics(hidden: np.ndarray, values: np.ndarray, groups: np.ndarray):
    y_reg = np.log1p(values)
    y_cls = (values >= 2.0).astype(int)
    n_layers = hidden.shape[1]
    n_splits = min(4, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)
    metrics = []
    for layer in range(n_layers):
        x = hidden[:, layer, :]
        pred_reg = np.zeros_like(y_reg, dtype=np.float64)
        pred_cls = np.zeros_like(y_cls, dtype=np.int64)
        for train_idx, test_idx in cv.split(x, y_reg, groups):
            reg = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
            reg.fit(x[train_idx], y_reg[train_idx])
            pred_reg[test_idx] = reg.predict(x[test_idx])
            clf = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("logreg", LogisticRegression(C=0.5, max_iter=1000, solver="liblinear")),
                ]
            )
            clf.fit(x[train_idx], y_cls[train_idx])
            pred_cls[test_idx] = clf.predict(x[test_idx])
        corr = float(np.corrcoef(y_reg, pred_reg)[0, 1])
        metrics.append(
            {
                "layer": int(layer),
                "r2_log_value": float(1.0 - np.sum((y_reg - pred_reg) ** 2) / np.sum((y_reg - np.mean(y_reg)) ** 2)),
                "corr_log_value": corr,
                "long_value_accuracy": float(accuracy_score(y_cls, pred_cls)),
            }
        )
    return metrics


def fit_direction(hidden_layer: np.ndarray, values: np.ndarray):
    y = np.log1p(values)
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    pipe.fit(hidden_layer, y)
    scaler = pipe.named_steps["scale"]
    ridge = pipe.named_steps["ridge"]
    coef = ridge.coef_ / scaler.scale_
    norm = np.linalg.norm(coef)
    direction = (coef / max(norm, 1e-12)).astype(np.float32)
    return pipe, direction


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def add_vector_hook(block, prompt_len: int, vector: torch.Tensor):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, None
        idx = min(prompt_len - 1, hidden.shape[1] - 1)
        patched = hidden.clone()
        patched[:, idx, :] = patched[:, idx, :] + vector.to(patched.device, patched.dtype)
        return (patched,) + rest if rest is not None else patched

    return block.register_forward_hook(hook)


def conditional_label_scores(tokenizer, model, prompt: str, layer: int, vector: np.ndarray):
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
    prompt_len = int(prompt_ids.shape[0])
    hook_vec = torch.tensor(vector, dtype=torch.float32).view(1, 1, -1)
    scores = {}
    blocks = get_blocks(model)
    for label in LABELS:
        cand_ids = tokenizer(" " + label, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        input_ids = torch.cat([prompt_ids, cand_ids], dim=0).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        handle = add_vector_hook(blocks[layer], prompt_len, hook_vec)
        with torch.inference_mode():
            out = model(input_ids=input_ids, labels=labels)
        handle.remove()
        scores[label] = float(-out.loss.detach().float().cpu().item())
    return scores


def winner(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def train_adapter(seconds: np.ndarray, deltas: np.ndarray, groups: np.ndarray, val_group_cutoff: int):
    hidden_size = deltas.shape[1]
    adapter = TimeAdapter(hidden_size)
    train_mask = groups < val_group_cutoff
    val_mask = ~train_mask
    x_train = torch.tensor(seconds[train_mask], dtype=torch.float32)
    y_train = torch.tensor(deltas[train_mask], dtype=torch.float32)
    x_val = torch.tensor(seconds[val_mask], dtype=torch.float32)
    y_val = torch.tensor(deltas[val_mask], dtype=torch.float32)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=1e-4)
    losses = []
    for epoch in range(350):
        adapter.train()
        opt.zero_grad()
        pred = adapter(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == 349:
            adapter.eval()
            with torch.no_grad():
                val_pred = adapter(x_val)
                val_loss = torch.mean((val_pred - y_val) ** 2)
                zero = torch.mean(y_val**2)
            losses.append(
                {
                    "epoch": epoch,
                    "train_mse": float(loss.detach().cpu()),
                    "val_mse": float(val_loss.detach().cpu()),
                    "val_zero_baseline_mse": float(zero.detach().cpu()),
                }
            )
    adapter.eval()
    with torch.no_grad():
        pred_np = adapter(x_val).detach().cpu().numpy()
    true_np = deltas[val_mask]
    denom = np.linalg.norm(pred_np, axis=1) * np.linalg.norm(true_np, axis=1)
    cosines = np.divide(np.sum(pred_np * true_np, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    return adapter, losses, {
        "val_mse": losses[-1]["val_mse"],
        "val_zero_baseline_mse": losses[-1]["val_zero_baseline_mse"],
        "val_mean_cosine": float(np.mean(cosines)),
        "val_median_cosine": float(np.median(cosines)),
    }


def make_time_examples():
    examples = []
    for variant_name, spec in TIME_VARIANTS.items():
        for fid, fragment in enumerate(FRAGMENTS):
            for seconds in TIMES:
                cue = spec["text"](seconds)
                examples.append(
                    HiddenExample(
                        variant=variant_name,
                        fragment_id=fid,
                        fragment=fragment,
                        value=seconds,
                        cue=cue,
                        prompt=build_prompt(fragment, cue),
                    )
                )
    return examples


def make_non_time_examples():
    examples = []
    for variant_name, spec in NON_TIME_VARIANTS.items():
        for fid, fragment in enumerate(FRAGMENTS):
            for value in CONTROL_VALUES:
                cue = spec["text"](value)
                examples.append(
                    HiddenExample(
                        variant=variant_name,
                        fragment_id=fid,
                        fragment=fragment,
                        value=value,
                        cue=cue,
                        prompt=build_control_prompt(fragment, cue),
                    )
                )
    return examples


CONTEXT_CASES = [
    {
        "id": "neutral_school_4s",
        "fragment": "今日学校で……",
        "seconds": 4.0,
        "gold": "SUPPORT",
        "acceptable": ["BACKCHANNEL", "SUPPORT"],
        "reason": "neutral incomplete story plus long pause",
    },
    {
        "id": "asked_wait_4s",
        "fragment": "ちょっと考えるから待って",
        "seconds": 4.0,
        "gold": "WAIT",
        "acceptable": ["WAIT"],
        "reason": "speaker explicitly asks for waiting",
    },
    {
        "id": "finished_05s",
        "fragment": "以上です",
        "seconds": 0.5,
        "gold": "SUPPORT",
        "acceptable": ["SUPPORT"],
        "reason": "turn appears complete even with short pause",
    },
    {
        "id": "hesitant_1s",
        "fragment": "えっと……どう言えばいいんだろう",
        "seconds": 1.0,
        "gold": "BACKCHANNEL",
        "acceptable": ["BACKCHANNEL"],
        "reason": "hesitation with moderate pause",
    },
    {
        "id": "summary_4s",
        "fragment": "つまり結論としては……",
        "seconds": 4.0,
        "gold": "BACKCHANNEL",
        "acceptable": ["WAIT", "BACKCHANNEL"],
        "reason": "speaker is likely preparing conclusion",
    },
    {
        "id": "vulnerable_2s",
        "fragment": "ごめん、ちょっと言いにくくて……",
        "seconds": 2.0,
        "gold": "SUPPORT",
        "acceptable": ["BACKCHANNEL", "SUPPORT"],
        "reason": "vulnerable content plus pause",
    },
]


CONTEXT_PROFILES = [
    ("neutral_incomplete", "今日学校で……", {0.5: "WAIT", 1.0: "BACKCHANNEL", 2.0: "BACKCHANNEL", 4.0: "SUPPORT", 8.0: "SUPPORT"}),
    ("asked_wait", "ちょっと考えるから待って", {0.5: "WAIT", 1.0: "WAIT", 2.0: "WAIT", 4.0: "WAIT", 8.0: "BACKCHANNEL"}),
    ("finished", "以上です", {0.5: "SUPPORT", 1.0: "SUPPORT", 2.0: "SUPPORT", 4.0: "SUPPORT", 8.0: "SUPPORT"}),
    ("hesitant", "えっと……どう言えばいいんだろう", {0.5: "WAIT", 1.0: "BACKCHANNEL", 2.0: "BACKCHANNEL", 4.0: "SUPPORT", 8.0: "SUPPORT"}),
    ("summary", "つまり結論としては……", {0.5: "WAIT", 1.0: "WAIT", 2.0: "BACKCHANNEL", 4.0: "BACKCHANNEL", 8.0: "SUPPORT"}),
    ("vulnerable", "ごめん、ちょっと言いにくくて……", {0.5: "BACKCHANNEL", 1.0: "BACKCHANNEL", 2.0: "SUPPORT", 4.0: "SUPPORT", 8.0: "SUPPORT"}),
]


def evaluate_contexts(tokenizer, model, layer, adapter, primary_variant="seconds_bracket"):
    rows = []
    profiles = []
    with torch.no_grad():
        for profile_id, fragment, gold_by_time in CONTEXT_PROFILES:
            for seconds, gold in gold_by_time.items():
                cue = TIME_VARIANTS[primary_variant]["text"](seconds)
                explicit_prompt = build_prompt(fragment, cue)
                base_prompt = build_prompt(fragment, None)
                explicit_scores = conditional_label_scores(tokenizer, model, explicit_prompt, layer, np.zeros(model.config.hidden_size, dtype=np.float32))
                vector = adapter(torch.tensor([seconds], dtype=torch.float32))[0].detach().cpu().numpy()
                adapter_scores = conditional_label_scores(tokenizer, model, base_prompt, layer, vector)
                explicit_pred = winner(explicit_scores)
                adapter_pred = winner(adapter_scores)
                profiles.append(
                    {
                        "profile_id": profile_id,
                        "fragment": fragment,
                        "seconds": seconds,
                        "gold": gold,
                        "explicit_scores": explicit_scores,
                        "explicit_pred": explicit_pred,
                        "adapter_scores": adapter_scores,
                        "adapter_pred": adapter_pred,
                    }
                )
        for case in CONTEXT_CASES:
            cue = TIME_VARIANTS[primary_variant]["text"](case["seconds"])
            explicit_prompt = build_prompt(case["fragment"], cue)
            base_prompt = build_prompt(case["fragment"], None)
            explicit_scores = conditional_label_scores(tokenizer, model, explicit_prompt, layer, np.zeros(model.config.hidden_size, dtype=np.float32))
            vector = adapter(torch.tensor([case["seconds"]], dtype=torch.float32))[0].detach().cpu().numpy()
            adapter_scores = conditional_label_scores(tokenizer, model, base_prompt, layer, vector)
            rows.append(
                {
                    **case,
                    "explicit_scores": explicit_scores,
                    "explicit_pred": winner(explicit_scores),
                    "adapter_scores": adapter_scores,
                    "adapter_pred": winner(adapter_scores),
                }
            )
    return rows, profiles


def classification_metrics(rows, pred_key="adapter_pred"):
    y_true = [r["gold"] for r in rows]
    y_pred = [r[pred_key] for r in rows]
    acceptable = [r[pred_key] in r.get("acceptable", [r["gold"]]) for r in rows]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "acceptable_accuracy": float(np.mean(acceptable)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def plot_cosine_matrix(names, matrix, path, title):
    plt.figure(figsize=(9, 7))
    im = plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Cosine similarity")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_context_heatmap(rows, path, score_source="adapter_scores", title="Context x time logits"):
    profiles = sorted(set(r["profile_id"] for r in rows))
    seconds = sorted(set(r["seconds"] for r in rows))
    data = np.zeros((len(profiles), len(seconds)))
    labels = []
    for i, prof in enumerate(profiles):
        labels.append(prof)
        for j, sec in enumerate(seconds):
            row = next(r for r in rows if r["profile_id"] == prof and r["seconds"] == sec)
            data[i, j] = row[score_source]["SUPPORT"] - row[score_source]["WAIT"]
    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(data, aspect="auto", cmap="PiYG")
    plt.colorbar(im, label="SUPPORT logprob - WAIT logprob")
    plt.xticks(range(len(seconds)), [str(s) for s in seconds])
    plt.yticks(range(len(profiles)), labels)
    plt.xlabel("Seconds")
    plt.title(title)
    for i in range(len(profiles)):
        for j in range(len(seconds)):
            plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(cm, labels, path, title):
    plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(np.array(cm), cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--output-dir", default="artifacts/qwen3_additional_validation")
    parser.add_argument("--figure-dir", default="output/figures/qwen3_additional_validation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if args.resume and summary_path.exists():
        print(f"Existing summary found: {summary_path}")
        return

    tokenizer, model, device = load_model(Path(args.cache_dir))
    hidden_size = model.config.hidden_size

    time_examples = make_time_examples()
    base_prompts = [build_prompt(fragment, None) for fragment in FRAGMENTS]
    base_hidden = extract_hidden(tokenizer, model, device, base_prompts)
    time_hidden = extract_hidden(tokenizer, model, device, [ex.prompt for ex in time_examples])
    non_time_examples = make_non_time_examples()
    non_time_hidden = extract_hidden(tokenizer, model, device, [ex.prompt for ex in non_time_examples])

    time_results = {}
    directions = {}
    adapters = {}
    for variant_name in TIME_VARIANTS:
        idxs = [i for i, ex in enumerate(time_examples) if ex.variant == variant_name]
        hidden = time_hidden[idxs]
        values = np.array([time_examples[i].value for i in idxs], dtype=np.float32)
        groups = np.array([time_examples[i].fragment_id for i in idxs], dtype=np.int64)
        metrics = cv_metrics(hidden, values, groups)
        best = max(metrics, key=lambda m: (m["r2_log_value"], m["long_value_accuracy"]))
        layer = int(best["layer"])
        _, direction = fit_direction(hidden[:, layer, :], values)
        directions[variant_name] = {"layer": layer, "direction": direction, "metrics": metrics, "best": best}

        deltas = []
        seconds = []
        delta_groups = []
        for i in idxs:
            ex = time_examples[i]
            deltas.append(time_hidden[i, layer, :] - base_hidden[ex.fragment_id, layer, :])
            seconds.append(ex.value)
            delta_groups.append(ex.fragment_id)
        adapter, losses, adapter_metrics = train_adapter(np.array(seconds), np.stack(deltas), np.array(delta_groups), val_group_cutoff=9)
        adapters[variant_name] = adapter
        time_results[variant_name] = {
            "best_layer": layer,
            "best_metrics": best,
            "adapter_metrics": adapter_metrics,
            "adapter_losses": losses,
        }
        torch.save(adapter.state_dict(), out_dir / f"adapter_{variant_name}.pt")
        np.save(out_dir / f"direction_{variant_name}.npy", direction)

    primary_layer = directions["seconds_bracket"]["layer"]
    primary_direction = directions["seconds_bracket"]["direction"]
    primary_adapter = adapters["seconds_bracket"]

    non_time_results = {}
    non_time_directions = {}
    for variant_name in NON_TIME_VARIANTS:
        idxs = [i for i, ex in enumerate(non_time_examples) if ex.variant == variant_name]
        hidden = non_time_hidden[idxs]
        values = np.array([non_time_examples[i].value for i in idxs], dtype=np.float32)
        groups = np.array([non_time_examples[i].fragment_id for i in idxs], dtype=np.int64)
        metrics = cv_metrics(hidden, values, groups)
        best_at_primary = metrics[primary_layer]
        _, direction = fit_direction(hidden[:, primary_layer, :], values)
        non_time_directions[variant_name] = direction
        np.save(out_dir / f"direction_non_time_{variant_name}.npy", direction)
        non_time_results[variant_name] = {
            "metrics_at_primary_layer": best_at_primary,
            "best_metrics": max(metrics, key=lambda m: (m["r2_log_value"], m["long_value_accuracy"])),
            "cosine_to_primary_time_direction": cosine(direction, primary_direction),
        }

    all_direction_names = list(TIME_VARIANTS.keys()) + [f"non_time_{k}" for k in NON_TIME_VARIANTS]
    all_vectors = [directions[n]["direction"] for n in TIME_VARIANTS] + [non_time_directions[k] for k in NON_TIME_VARIANTS]
    cos_matrix = [[cosine(a, b) for b in all_vectors] for a in all_vectors]
    plot_cosine_matrix(all_direction_names, cos_matrix, fig_dir / "direction_cosine_matrix.png", "Direction cosine similarities")

    # Direction and adapter interventions on no-time prompts.
    probe_fragments = FRAGMENTS[:6]
    intervention_seconds = [0.5, 1.0, 2.0, 4.0, 8.0]
    intervention = {"time_variants": {}, "non_time": {}, "random": []}
    zero = np.zeros(hidden_size, dtype=np.float32)

    for variant_name, adapter in adapters.items():
        rows = []
        explicit_rows = []
        for seconds in intervention_seconds:
            adapter_scores_all = []
            explicit_scores_all = []
            with torch.no_grad():
                vector = adapter(torch.tensor([seconds], dtype=torch.float32))[0].detach().cpu().numpy()
            for frag in probe_fragments:
                base_prompt = build_prompt(frag, None)
                cue = TIME_VARIANTS[variant_name]["text"](seconds)
                explicit_prompt = build_prompt(frag, cue)
                adapter_scores_all.append(conditional_label_scores(tokenizer, model, base_prompt, directions[variant_name]["layer"], vector))
                explicit_scores_all.append(conditional_label_scores(tokenizer, model, explicit_prompt, directions[variant_name]["layer"], zero))
            adapter_mean = {lab: float(np.mean([s[lab] for s in adapter_scores_all])) for lab in LABELS}
            explicit_mean = {lab: float(np.mean([s[lab] for s in explicit_scores_all])) for lab in LABELS}
            rows.append({"seconds": seconds, "mean_scores": adapter_mean, "winner": winner(adapter_mean)})
            explicit_rows.append({"seconds": seconds, "mean_scores": explicit_mean, "winner": winner(explicit_mean)})
        agreement = float(np.mean([a["winner"] == e["winner"] for a, e in zip(rows, explicit_rows)]))
        intervention["time_variants"][variant_name] = {
            "adapter": rows,
            "explicit": explicit_rows,
            "explicit_adapter_boundary_agreement": agreement,
        }

    # Match direction norm/scale from primary adapter at each second for non-time and random baselines.
    rng = np.random.default_rng(SEED)
    for variant_name, direction in non_time_directions.items():
        rows = []
        for value in intervention_seconds:
            with torch.no_grad():
                ref_vector = primary_adapter(torch.tensor([value], dtype=torch.float32))[0].detach().cpu().numpy()
            vector = direction * np.linalg.norm(ref_vector)
            score_rows = []
            for frag in probe_fragments:
                score_rows.append(conditional_label_scores(tokenizer, model, build_prompt(frag, None), primary_layer, vector))
            mean_scores = {lab: float(np.mean([s[lab] for s in score_rows])) for lab in LABELS}
            rows.append({"value": value, "mean_scores": mean_scores, "winner": winner(mean_scores)})
        intervention["non_time"][variant_name] = rows

    for trial in range(5):
        rand = rng.normal(size=hidden_size).astype(np.float32)
        rand = rand / max(np.linalg.norm(rand), 1e-12)
        rows = []
        for value in intervention_seconds:
            with torch.no_grad():
                ref_vector = primary_adapter(torch.tensor([value], dtype=torch.float32))[0].detach().cpu().numpy()
            vector = rand * np.linalg.norm(ref_vector)
            score_rows = []
            for frag in probe_fragments:
                score_rows.append(conditional_label_scores(tokenizer, model, build_prompt(frag, None), primary_layer, vector))
            mean_scores = {lab: float(np.mean([s[lab] for s in score_rows])) for lab in LABELS}
            rows.append({"value": value, "mean_scores": mean_scores, "winner": winner(mean_scores)})
        intervention["random"].append({"trial": trial, "rows": rows})

    context_cases, context_profiles = evaluate_contexts(tokenizer, model, primary_layer, primary_adapter)
    context_case_metrics_adapter = classification_metrics(context_cases, "adapter_pred")
    context_case_metrics_explicit = classification_metrics(context_cases, "explicit_pred")
    context_profile_metrics_adapter = classification_metrics(context_profiles, "adapter_pred")
    context_profile_metrics_explicit = classification_metrics(context_profiles, "explicit_pred")
    plot_context_heatmap(context_profiles, fig_dir / "context_time_adapter_heatmap.png", title="Context x time: adapter SUPPORT-WAIT")
    plot_context_heatmap(context_profiles, fig_dir / "context_time_explicit_heatmap.png", score_source="explicit_scores", title="Context x time: explicit SUPPORT-WAIT")
    plot_confusion(context_profile_metrics_adapter["confusion_matrix"], LABELS, fig_dir / "context_adapter_confusion.png", "Adapter context evaluation")
    plot_confusion(context_profile_metrics_explicit["confusion_matrix"], LABELS, fig_dir / "context_explicit_confusion.png", "Explicit context evaluation")

    # Save compact logit tables.
    logit_tables = {
        "context_profiles": [
            {
                "profile_id": r["profile_id"],
                "seconds": r["seconds"],
                "gold": r["gold"],
                "explicit_pred": r["explicit_pred"],
                "adapter_pred": r["adapter_pred"],
                "explicit_scores": r["explicit_scores"],
                "adapter_scores": r["adapter_scores"],
            }
            for r in context_profiles
        ],
        "time_variant_intervention": intervention["time_variants"],
        "non_time_intervention": intervention["non_time"],
        "random_intervention": intervention["random"],
    }

    summary = {
        "model_id": MODEL_ID,
        "fragments": FRAGMENTS,
        "times": TIMES,
        "primary_layer": primary_layer,
        "time_variants": time_results,
        "time_direction_cosine_matrix": {"names": all_direction_names, "matrix": cos_matrix},
        "non_time_controls": non_time_results,
        "intervention": intervention,
        "context_cases": context_cases,
        "context_case_metrics_adapter": context_case_metrics_adapter,
        "context_case_metrics_explicit": context_case_metrics_explicit,
        "context_profiles": context_profiles,
        "context_profile_metrics_adapter": context_profile_metrics_adapter,
        "context_profile_metrics_explicit": context_profile_metrics_explicit,
        "logit_tables": logit_tables,
        "figures": {
            "direction_cosine_matrix": str(fig_dir / "direction_cosine_matrix.png"),
            "context_time_adapter_heatmap": str(fig_dir / "context_time_adapter_heatmap.png"),
            "context_time_explicit_heatmap": str(fig_dir / "context_time_explicit_heatmap.png"),
            "context_adapter_confusion": str(fig_dir / "context_adapter_confusion.png"),
            "context_explicit_confusion": str(fig_dir / "context_explicit_confusion.png"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
