import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/qwen3_context_time_phase1_3000"
SEED = 20260623
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]
SPLIT_SIZES = {"train": 3000, "validation": 500, "test": 500}
SPLIT_TIMES = {
    "train": [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0],
    "validation": [0.2, 0.5, 0.8, 1.5, 3.0, 6.0],
    "test": [0.3, 0.8, 1.5, 3.0, 6.0, 7.0],
}
PROFILES = [
    "neutral_incomplete",
    "asked_wait",
    "finished",
    "hesitant",
    "summary",
    "vulnerable",
    "self_repair",
    "direct_question",
]


TOPIC_SEEDS = [
    "今日の授業で先生に言われたこと",
    "昨日の帰り道に友だちと話したこと",
    "朝の会議で急に振られた話",
    "家族に送ったメッセージの返事",
    "病院で聞いた検査結果",
    "部活の後に先輩から言われた一言",
    "駅で偶然会った人との会話",
    "提出した資料に入れ忘れた説明",
    "アルバイト先で起きた小さな失敗",
    "オンライン面談で言いそびれたこと",
    "友人から相談された内容",
    "週末に見たニュースのこと",
    "ゼミで発表したときの反応",
    "新しい仕事の引き継ぎ",
    "予約の電話で聞かれたこと",
    "買い物の途中で気づいた違和感",
    "昔の同級生から届いた連絡",
    "面接で答えに詰まった質問",
    "旅行の予定を立てていたときの話",
    "会計の数字が合わなかった件",
    "チーム内で共有したかった懸念",
    "授業後に残って聞いた質問",
    "上司に確認したかった判断",
    "帰宅してから思い出した約束",
    "長いメールを書いていた理由",
    "急に予定が変わった日のこと",
    "先生に褒められたけれど気になった点",
    "友だちの表情が少し暗かったこと",
    "締め切り前に迷っていた選択",
    "新しいアプリを試していたときのこと",
    "会議室で誰も話さなくなった瞬間",
    "親に言い出せなかった相談",
    "レビューで指摘された箇所",
    "帰り道に考えていた将来の話",
    "相手の返事が遅れて不安になったこと",
    "説明会で聞いた条件",
    "財布を忘れたかもしれないと思った瞬間",
    "昨日読んだ本で引っかかった言葉",
    "発表前に緊張していた理由",
    "久しぶりに会った人に言われたこと",
    "チケットを取ろうとして失敗した話",
    "練習中にうまくいかなかった動き",
    "相談窓口に行こうか迷った件",
    "朝から頭に残っている出来事",
    "相手に謝りたいと思っている理由",
    "予定表を見直して気づいた問題",
    "資料の最後に入れたい補足",
    "プレゼントを選んでいたときの迷い",
    "店員さんに説明された注意点",
    "昨日の夜に眠れなかった理由",
    "電話を切ったあとで気づいたこと",
    "同僚が急に黙った場面",
    "試験が終わったあとの気持ち",
    "次の面談で聞きたいこと",
    "メモに残した一文の意味",
    "短い返信だけでは伝わらなかったこと",
    "少し距離を置きたいと思った理由",
    "相手の冗談が気になっていること",
    "会議の結論として伝えるべき点",
    "今朝から何度も考えていること",
]

TOPIC_MODIFIERS = [
    "一つ目の場面",
    "別の日の似た場面",
    "少し前に起きた場面",
    "相手が急いでいた場面",
    "こちらが緊張していた場面",
    "相手に悪気がなさそうな場面",
    "誰にもまだ相談していない場面",
    "後から気になってきた場面",
    "説明を途中で止めた場面",
    "言葉を選んでいた場面",
    "結論だけ言いかけた場面",
    "気持ちが揺れていた場面",
]

TEMPLATES = {
    "neutral_incomplete": [
        "{topic}なんだけど……",
        "{topic}について話そうとしていて……",
        "{topic}で少し引っかかっていて……",
        "{topic}の続きなんだけど……",
        "{topic}を思い出していて……",
        "{topic}を説明すると、まず……",
        "{topic}で、そこから先が……",
        "{topic}について、まだ途中なんだけど……",
    ],
    "asked_wait": [
        "{topic}を整理したいから、少し待って",
        "{topic}について考える時間がほしい",
        "{topic}は今まとめているところだから待って",
        "{topic}の順番を確認するから、まだ聞いていて",
        "{topic}を間違えたくないので少し待って",
        "{topic}は言葉を選びたいから待って",
        "{topic}は急かさないで聞いてほしい",
        "{topic}について、あと少しだけ考えさせて",
    ],
    "finished": [
        "{topic}については以上です",
        "{topic}はこれで全部です",
        "{topic}の説明はここまでです",
        "{topic}について伝えたいことは終わりました",
        "{topic}は今ので結論です",
        "{topic}はそんな感じです",
        "{topic}については、これで答えです",
        "{topic}の話はここで一区切りです",
    ],
    "hesitant": [
        "{topic}のこと、えっと……どう言えばいいんだろう",
        "{topic}について、うまく言えないんだけど……",
        "{topic}を話すのは少し難しくて……",
        "{topic}で、ええと、言葉が出てこなくて……",
        "{topic}の説明、ちょっと迷っていて……",
        "{topic}について、まだ整理できていなくて……",
        "{topic}のこと、言い始めたけど詰まってしまって……",
        "{topic}は、なんて言うのが近いかな……",
    ],
    "summary": [
        "{topic}の結論としては……",
        "{topic}をまとめると……",
        "{topic}で一番言いたいのは……",
        "{topic}について要するに……",
        "{topic}のポイントはたぶん……",
        "{topic}を一言で言うなら……",
        "{topic}で最後に言うべきなのは……",
        "{topic}について結局のところ……",
    ],
    "vulnerable": [
        "{topic}のことで、ごめん、少し言いにくくて……",
        "{topic}について、正直ちょっと怖くて……",
        "{topic}を話すのが恥ずかしいんだけど……",
        "{topic}で、責められるかもしれないと思って……",
        "{topic}について、弱音みたいで言いづらいけど……",
        "{topic}のこと、本当は助けてほしくて……",
        "{topic}で、笑われるかもしれないけど……",
        "{topic}について、一人で抱えるのがしんどくて……",
    ],
    "self_repair": [
        "{topic}について、いや違う、そうじゃなくて……",
        "{topic}は、あ、今の言い方だと変かもしれない……",
        "{topic}を説明すると、いや順番を直すと……",
        "{topic}で、待って、言い直すね……",
        "{topic}について、少し訂正すると……",
        "{topic}は、今の表現を変えるなら……",
        "{topic}の話、さっきの言い方を直すと……",
        "{topic}について、違う言い方にすると……",
    ],
    "direct_question": [
        "{topic}について、どう思いますか？",
        "{topic}の場合、あなたならどうしますか？",
        "{topic}について意見を聞かせてください",
        "{topic}はどう受け止めればいいですか？",
        "{topic}で次に何をすればいいでしょうか？",
        "{topic}について助言をもらえますか？",
        "{topic}はどう返せばよかったですか？",
        "{topic}について判断を手伝ってください",
    ],
}


def label_for(profile: str, seconds: float) -> tuple[str, list[str], str]:
    if profile == "asked_wait":
        if seconds >= 7.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"], "長すぎる沈黙では待つ意図を尊重しつつ短い相づちも自然"
        return "WAIT", ["WAIT"], "話者が明示的に待つよう依頼している"
    if profile == "finished":
        return "SUPPORT", ["SUPPORT"], "発話が完了しているため短い沈黙でも応答してよい"
    if profile == "direct_question":
        return "SUPPORT", ["SUPPORT"], "明示的に回答や助言を求めている"
    if profile == "vulnerable":
        if seconds < 0.5:
            return "BACKCHANNEL", ["BACKCHANNEL", "SUPPORT"], "言いづらさへの短い受け止めが自然"
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"], "脆弱な内容なので支援的に応答してよい"
    if profile == "hesitant":
        if seconds < 0.5:
            return "WAIT", ["WAIT"], "言葉を探し始めた直後"
        if seconds < 3.0:
            return "BACKCHANNEL", ["BACKCHANNEL"], "迷いに対する短い相づちが自然"
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"], "長い迷いには助け舟が許容"
    if profile == "summary":
        if seconds < 0.8:
            return "WAIT", ["WAIT"], "結論を言い始める前"
        if seconds < 6.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"], "まとめを待つか短く促す場面"
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"], "長い沈黙では応答へ移ってよい"
    if profile == "self_repair":
        if seconds < 1.0:
            return "WAIT", ["WAIT"], "言い直しの途中なので待つ"
        return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"], "言い直しを邪魔しない短い促し"
    if profile == "neutral_incomplete":
        if seconds < 0.5:
            return "WAIT", ["WAIT"], "普通の言い差しで沈黙が短い"
        if seconds < 2.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"], "短い相づちで継続を促す"
        return "SUPPORT", ["BACKCHANNEL", "SUPPORT"], "長い沈黙では助け舟が自然"
    raise ValueError(profile)


def soft_label_for(profile: str, seconds: float, gold: str, acceptable: list[str]) -> dict[str, float]:
    probs = {label: 0.04 for label in LABELS}
    if len(acceptable) == 1:
        probs[gold] = 0.92
    else:
        probs[gold] = 0.68
        for label in acceptable:
            if label != gold:
                probs[label] = 0.28

    # Boundary regions are intentionally softer.
    if profile == "neutral_incomplete" and seconds in [0.5, 1.0, 2.0, 3.0]:
        probs = {"WAIT": 0.18, "BACKCHANNEL": 0.44, "SUPPORT": 0.38}
    elif profile == "asked_wait" and seconds in [6.0, 7.0, 8.0]:
        probs = {"WAIT": 0.52, "BACKCHANNEL": 0.42, "SUPPORT": 0.06}
    elif profile == "self_repair" and seconds in [0.8, 1.0, 1.5, 2.0, 3.0]:
        probs = {"WAIT": 0.40, "BACKCHANNEL": 0.54, "SUPPORT": 0.06}
    elif profile == "vulnerable" and seconds in [0.3, 0.5, 0.8]:
        probs = {"WAIT": 0.06, "BACKCHANNEL": 0.44, "SUPPORT": 0.50}
    elif profile == "hesitant" and seconds in [0.8, 1.0, 1.5, 2.0, 3.0]:
        probs = {"WAIT": 0.12, "BACKCHANNEL": 0.55, "SUPPORT": 0.33}
    elif profile == "summary" and seconds in [0.8, 1.0, 1.5, 3.0, 6.0]:
        probs = {"WAIT": 0.24, "BACKCHANNEL": 0.50, "SUPPORT": 0.26}

    total = sum(probs.values())
    return {label: round(probs[label] / total, 6) for label in LABELS}


def natural_time(seconds: float, variant: int) -> str:
    phrases = {
        0.0: ["すぐに続けて", "間を置かずに", "ほぼ沈黙なしで"],
        0.2: ["ごく短い沈黙のあと", "一瞬だけ黙って", "ほんの少し間を置いて"],
        0.3: ["ごく短く黙って", "一瞬弱の間を置いて", "短い息継ぎのあと"],
        0.5: ["短い沈黙のあと", "少しだけ黙って", "短く間を置いて"],
        0.8: ["やや短い沈黙のあと", "少し考えるように黙って", "ひと呼吸置いて"],
        1.0: ["1秒ほど黙って", "少し沈黙して", "短めの間を置いて"],
        1.5: ["1.5秒ほど沈黙して", "少し長めに黙って", "考える間を置いて"],
        2.0: ["2秒ほど沈黙して", "しばらく黙って", "少し長く間を置いて"],
        3.0: ["3秒ほど沈黙して", "長めに黙って", "はっきり間を置いて"],
        4.0: ["4秒ほど沈黙して", "長い沈黙のあと", "しばらく長く黙って"],
        6.0: ["6秒ほど沈黙して", "かなり長く黙って", "長い間を置いて"],
        7.0: ["7秒ほど沈黙して", "かなり長い沈黙のあと", "長く黙り込んで"],
        8.0: ["8秒ほど沈黙して", "とても長い沈黙のあと", "長く黙り込んだあと"],
    }
    return phrases[seconds][variant % 3]


def time_expression(seconds: float, index: int) -> tuple[str, str]:
    mode = index % 5
    if mode == 0:
        return f"[{seconds:g}s]", "seconds_bracket"
    if mode == 1:
        return f"[{int(round(seconds * 1000))}ms]", "milliseconds_bracket"
    if mode == 2:
        return f"{seconds:g}秒の沈黙", "seconds_japanese"
    if mode == 3:
        return natural_time(seconds, index), "natural_japanese"
    return f"沈黙時間は約{seconds:g}秒", "sentence_japanese"


def negative_control(index: int) -> tuple[bool, str | None]:
    if index % 6 != 0:
        return False, None
    values = [0.5, 1.0, 2.0, 5.0, 8.0]
    units = ["kg", "m", "点", "円"]
    value = values[(index // 6) % len(values)]
    unit = units[(index // 10) % len(units)]
    return True, f"[{value:g}{unit}]"


def make_topic(serial: int, split: str) -> str:
    seed = TOPIC_SEEDS[serial % len(TOPIC_SEEDS)]
    mod = TOPIC_MODIFIERS[(serial // len(TOPIC_SEEDS)) % len(TOPIC_MODIFIERS)]
    return f"{seed}（{split}:{mod}:{serial}）"


def make_fragment(profile: str, serial: int, split: str) -> str:
    topic = make_topic(serial, split)
    template = TEMPLATES[profile][(serial * 7 + len(profile)) % len(TEMPLATES[profile])]
    return template.format(topic=topic)


def make_row(split: str, idx: int, context_id: str, profile: str, fragment: str, seconds: float) -> dict:
    label, acceptable, rationale = label_for(profile, seconds)
    cue, cue_type = time_expression(seconds, idx)
    has_control, control_note = negative_control(idx)
    return {
        "id": f"{split}_{idx:05d}",
        "split": split,
        "context_group_id": context_id,
        "profile": profile,
        "fragment": fragment,
        "seconds": seconds,
        "time_expression": cue,
        "time_expression_type": cue_type,
        "label": label,
        "soft_label": soft_label_for(profile, seconds, label, acceptable),
        "acceptable_labels": acceptable,
        "rationale": rationale,
        "has_negative_control": has_control,
        "unrelated_numeric_note": control_note,
    }


def build_split(split: str, size: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    times = SPLIT_TIMES[split]
    split_offset = {"train": 0, "validation": 100_000, "test": 200_000}[split]
    profile_targets = {profile: size // len(PROFILES) for profile in PROFILES}
    for profile in PROFILES[: size % len(PROFILES)]:
        profile_targets[profile] += 1
    profile_counts = Counter()

    # Transition bundles: same fragment/context group appears at multiple time points.
    transition_profiles = ["neutral_incomplete", "asked_wait", "hesitant", "summary", "vulnerable", "self_repair"]
    bundle_times = times[:]
    serial = split_offset
    for profile in transition_profiles:
        bundle_count = max(1, int(profile_targets[profile] * 0.45 / len(bundle_times)))
        for _ in range(bundle_count):
            if profile_counts[profile] + len(bundle_times) > profile_targets[profile]:
                break
            fragment = make_fragment(profile, serial, split)
            context_id = f"{split}_ctx_{serial:06d}"
            for seconds in bundle_times:
                rows.append(make_row(split, len(rows), context_id, profile, fragment, seconds))
                profile_counts[profile] += 1
            serial += 1

    # Fill remaining rows to keep profiles balanced. Time values are cycled per profile.
    time_offsets = {profile: (i * 2 + len(profile)) % len(times) for i, profile in enumerate(PROFILES)}
    while len(rows) < size:
        profile = min(PROFILES, key=lambda p: (profile_counts[p] / profile_targets[p], profile_counts[p]))
        if profile_counts[profile] >= profile_targets[profile]:
            break
        seconds = times[(profile_counts[profile] + time_offsets[profile]) % len(times)]
        serial += 1
        fragment = make_fragment(profile, serial, split)
        context_id = f"{split}_ctx_{serial:06d}"
        rows.append(make_row(split, len(rows), context_id, profile, fragment, seconds))
        profile_counts[profile] += 1

    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        row["id"] = f"{split}_{idx:05d}"
    return rows[:size]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_overlap(all_rows: dict[str, list[dict]]) -> dict[str, int]:
    fragments = {split: {row["fragment"] for row in rows} for split, rows in all_rows.items()}
    return {
        "train_validation": len(fragments["train"] & fragments["validation"]),
        "train_test": len(fragments["train"] & fragments["test"]),
        "validation_test": len(fragments["validation"] & fragments["test"]),
    }


def transition_group_counts(rows: list[dict]) -> dict:
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["context_group_id"]].append(row)
    label_changing = 0
    multi_time = 0
    for group_rows in by_group.values():
        if len({row["seconds"] for row in group_rows}) > 1:
            multi_time += 1
        if len({row["label"] for row in group_rows}) > 1:
            label_changing += 1
    return {"multi_time_context_groups": multi_time, "label_changing_context_groups": label_changing}


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = {split: build_split(split, size, rng) for split, size in SPLIT_SIZES.items()}
    for split, rows in all_rows.items():
        write_jsonl(OUT_DIR / f"{split}.jsonl", rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "labels": LABELS,
        "split_sizes": {split: len(rows) for split, rows in all_rows.items()},
        "split_times": SPLIT_TIMES,
        "train_times": SPLIT_TIMES["train"],
        "validation_unseen_times": [t for t in SPLIT_TIMES["validation"] if t not in SPLIT_TIMES["train"]],
        "test_unseen_times": [t for t in SPLIT_TIMES["test"] if t not in SPLIT_TIMES["train"]],
        "profiles": PROFILES,
        "label_counts": {split: dict(Counter(row["label"] for row in rows)) for split, rows in all_rows.items()},
        "seconds_label_counts": {
            split: {str(sec): dict(Counter(row["label"] for row in rows if row["seconds"] == sec)) for sec in sorted({r["seconds"] for r in rows})}
            for split, rows in all_rows.items()
        },
        "profile_counts": {split: dict(Counter(row["profile"] for row in rows)) for split, rows in all_rows.items()},
        "time_expression_counts": {
            split: dict(Counter(row["time_expression_type"] for row in rows)) for split, rows in all_rows.items()
        },
        "negative_control_counts": {
            split: sum(1 for row in rows if row["has_negative_control"]) for split, rows in all_rows.items()
        },
        "split_fragment_overlap": split_overlap(all_rows),
        "transition_group_counts": {split: transition_group_counts(rows) for split, rows in all_rows.items()},
        "notes": [
            "Exact fragment overlap across train/validation/test is zero.",
            "Within each split, transition bundles intentionally repeat a context_group_id at multiple time points to test label transitions for the same context.",
            "Labels are context-dependent; each seconds value has multiple labels.",
            "Soft labels are included for boundary and acceptable-label ambiguity.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {OUT_DIR}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
