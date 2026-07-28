from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "expanded_v4_clean": ROOT / "data/qwen3_context_time_expanded",
    "phase1_v5_clean": ROOT / "data/qwen3_context_time_phase1_3000",
}
OUTS = {
    "expanded_v4_clean": ROOT / "data/qwen3_context_time_expanded_clean_revalidation_v1",
    "phase1_v5_clean": ROOT / "data/qwen3_context_time_phase1_3000_clean_revalidation_v1",
}
SPLITS = ["train", "validation", "test"]
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]

FW_L = "\uff08"
FW_R = "\uff09"
BETSU_MARKER = "\u5225\u306e\u5834\u9762\u3068\u3057\u3066"

QUALIFIER_A = [
    "前日の出来事を思い出しながら",
    "帰り道で順番を確認しながら",
    "メモを見返しながら",
    "記録を見直しながら",
    "関連する出来事を思い出しながら",
    "少し整理し直しながら",
    "手元の内容を確認しながら",
    "その時の流れを追いながら",
    "あとから気になった点を考えながら",
    "前後の流れを確認しながら",
    "状況を思い返しながら",
    "出来事の順番を思い返しながら",
]

QUALIFIER_B = [
    "もう一度説明すると",
    "別の角度から説明すると",
    "なるべく正確に言うと",
    "少し補足すると",
    "別の言い方にすると",
    "思い出せる範囲で言うと",
    "大事なところだけ言うと",
    "落ち着いて言い直すと",
    "その後のことも含めると",
    "短くまとめると",
    "順序を入れ替えて言うと",
    "順番に言うと",
]

QUALIFIER_C = [
    "朝のこととして",
    "帰る前のこととして",
    "会話の途中のこととして",
    "少し時間を置いた後のこととして",
    "別の日に思い返したこととして",
    "その場で考えたこととして",
    "落ち着いてから気づいたこととして",
    "あとで整理したこととして",
    "相手に伝え直すつもりで",
    "自分の中で確認しながら",
    "話の流れを戻すつもりで",
    "言い残しを拾うつもりで",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_index(text: str, n: int) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % n


def strip_markers(fragment: str) -> str:
    s = fragment

    def split_marker_repl(match: re.Match[str]) -> str:
        modifier = match.group(1)
        return f"の{modifier}"

    s = re.sub(FW_L + r"(?:train|validation|test):([^:]+):\d+" + FW_R, split_marker_repl, s)
    s = s.replace(FW_L + BETSU_MARKER + FW_R, "")
    s = re.sub(r"\s+\d{3,}\s*$", "", s)
    s = s.replace("  ", " ").strip()
    return s


def make_neutral_qualifier(key: str) -> str:
    a = QUALIFIER_A[stable_index(key + "a", len(QUALIFIER_A))]
    b = QUALIFIER_B[stable_index(key + "b", len(QUALIFIER_B))]
    c = QUALIFIER_C[stable_index(key + "c", len(QUALIFIER_C))]
    return f"{a}、{c}、{b}、"


def clean_rows(rows_by_split: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    group_to_fragment: dict[tuple[str, str], str] = {}
    used_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    all_used: set[str] = set()
    cleaned_by_split: dict[str, list[dict]] = {}
    changed = Counter()
    collision_fixes = Counter()

    for split in SPLITS:
        out_rows = []
        for idx, row in enumerate(rows_by_split[split]):
            row = dict(row)
            source_fragment = row["fragment"]
            stripped = strip_markers(source_fragment)
            group_id = row.get("context_group_id") or row["id"]
            group_key = (split, group_id)
            if group_key not in group_to_fragment:
                qualifier = make_neutral_qualifier(f"{split}:{group_id}:{stripped}")
                candidate = qualifier + stripped
                attempts = 0
                while candidate in all_used:
                    attempts += 1
                    qualifier = make_neutral_qualifier(f"{split}:{group_id}:{stripped}:retry:{attempts}")
                    candidate = qualifier + stripped
                group_to_fragment[group_key] = candidate
                if attempts:
                    collision_fixes[split] += 1
            clean_fragment = group_to_fragment[group_key]
            if clean_fragment != source_fragment:
                changed[split] += 1
            row["fragment_original"] = source_fragment
            row["fragment"] = clean_fragment
            row["cleaning_notes"] = {
                "removed_split_serial_marker": bool(re.search(FW_L + r"(?:train|validation|test):", source_fragment)),
                "removed_duplicate_marker": BETSU_MARKER in source_fragment,
                "added_neutral_qualifier": True,
            }
            out_rows.append(row)
            used_by_split[split].add(clean_fragment)
            all_used.add(clean_fragment)
        cleaned_by_split[split] = out_rows

    audit = audit_dataset(cleaned_by_split)
    audit["changed_counts"] = dict(changed)
    audit["collision_fixes"] = dict(collision_fixes)
    return cleaned_by_split, audit


def audit_dataset(rows_by_split: dict[str, list[dict]]) -> dict:
    out = {
        "split_sizes": {split: len(rows_by_split[split]) for split in SPLITS},
        "label_counts": {split: dict(Counter(row["label"] for row in rows_by_split[split])) for split in SPLITS},
        "profile_counts": {split: dict(Counter(row["profile"] for row in rows_by_split[split])) for split in SPLITS},
        "marker_counts": {},
        "fragment_overlap": {},
        "unique_fragments": {},
    }
    for split, rows in rows_by_split.items():
        fragments = [row["fragment"] for row in rows]
        out["unique_fragments"][split] = len(set(fragments))
        out["marker_counts"][split] = {
            "contains_split_colon": sum(any(x in f for x in ["train:", "validation:", "test:"]) for f in fragments),
            "contains_fullwidth_parentheses": sum(FW_L in f or FW_R in f for f in fragments),
            "contains_ascii_parentheses": sum("(" in f or ")" in f for f in fragments),
            "contains_long_digit_run": sum(bool(re.search(r"\d{3,}", f)) for f in fragments),
            "contains_duplicate_marker": sum(BETSU_MARKER in f for f in fragments),
        }
    frag_sets = {split: {row["fragment"] for row in rows_by_split[split]} for split in SPLITS}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        out["fragment_overlap"][f"{a}_{b}"] = len(frag_sets[a] & frag_sets[b])
    by_group = defaultdict(list)
    for split, rows in rows_by_split.items():
        for row in rows:
            by_group[(split, row.get("context_group_id", row["id"]))].append(row)
    out["transition_group_counts"] = {
        split: {
            "multi_time_context_groups": sum(1 for (s, _), rs in by_group.items() if s == split and len(rs) > 1),
            "label_changing_context_groups": sum(1 for (s, _), rs in by_group.items() if s == split and len({r["label"] for r in rs}) > 1),
        }
        for split in SPLITS
    }
    return out


def main():
    for name, source_dir in SOURCES.items():
        out_dir = OUTS[name]
        rows_by_split = {split: read_jsonl(source_dir / f"{split}.jsonl") for split in SPLITS}
        cleaned, audit = clean_rows(rows_by_split)
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in cleaned.items():
            write_jsonl(out_dir / f"{split}.jsonl", rows)
        source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest = {
            **source_manifest,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_dataset": str(source_dir),
            "clean_revalidation_version": 1,
            "cleaning_policy": [
                "Remove split names, serial ids, duplicate disambiguation markers, parentheses, and long numeric ids from fragment text.",
                "Add neutral Japanese qualifier phrases to preserve no-overlap constraints without visible IDs or tags.",
                "Keep label/profile/seconds/time expression/negative-control fields unchanged.",
            ],
            "audit": audit,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
