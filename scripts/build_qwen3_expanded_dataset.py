import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/qwen3_context_time_expanded"
SEED = 20260622
LABELS = ["WAIT", "BACKCHANNEL", "SUPPORT"]

SPLIT_SIZES = {"train": 500, "validation": 100, "test": 100}
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

TOPICS = [
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

TEMPLATES = {
    "neutral_incomplete": [
        "{topic}なんだけど……",
        "{topic}について話そうとしていて……",
        "{topic}で少し引っかかっていて……",
        "{topic}の続きなんだけど……",
        "{topic}を思い出していて……",
        "{topic}を説明すると、まず……",
    ],
    "asked_wait": [
        "{topic}を整理したいから、少し待って",
        "{topic}について考える時間がほしい",
        "{topic}は今まとめているところだから待って",
        "{topic}の順番を確認するから、まだ聞いていて",
        "{topic}を間違えたくないので少し待って",
        "{topic}は言葉を選びたいから待って",
    ],
    "finished": [
        "{topic}については以上です",
        "{topic}はこれで全部です",
        "{topic}の説明はここまでです",
        "{topic}について伝えたいことは終わりました",
        "{topic}は今ので結論です",
        "{topic}はそんな感じです",
    ],
    "hesitant": [
        "{topic}のこと、えっと……どう言えばいいんだろう",
        "{topic}について、うまく言えないんだけど……",
        "{topic}を話すのは少し難しくて……",
        "{topic}で、ええと、言葉が出てこなくて……",
        "{topic}の説明、ちょっと迷っていて……",
        "{topic}について、まだ整理できていなくて……",
    ],
    "summary": [
        "{topic}の結論としては……",
        "{topic}をまとめると……",
        "{topic}で一番言いたいのは……",
        "{topic}について要するに……",
        "{topic}のポイントはたぶん……",
        "{topic}を一言で言うなら……",
    ],
    "vulnerable": [
        "{topic}のことで、ごめん、少し言いにくくて……",
        "{topic}について、正直ちょっと怖くて……",
        "{topic}を話すのが恥ずかしいんだけど……",
        "{topic}で、責められるかもしれないと思って……",
        "{topic}について、弱音みたいで言いづらいけど……",
        "{topic}のこと、本当は助けてほしくて……",
    ],
    "self_repair": [
        "{topic}について、いや違う、そうじゃなくて……",
        "{topic}は、あ、今の言い方だと変かもしれない……",
        "{topic}を説明すると、いや順番を直すと……",
        "{topic}で、待って、言い直すね……",
        "{topic}について、少し訂正すると……",
        "{topic}は、今の表現を変えるなら……",
    ],
    "direct_question": [
        "{topic}について、どう思いますか？",
        "{topic}の場合、あなたならどうしますか？",
        "{topic}について意見を聞かせてください",
        "{topic}はどう受け止めればいいですか？",
        "{topic}で次に何をすればいいでしょうか？",
        "{topic}について助言をもらえますか？",
    ],
}


def label_for(profile: str, seconds: float) -> tuple[str, list[str], str]:
    if profile == "asked_wait":
        if seconds >= 7.0:
            return "BACKCHANNEL", ["WAIT", "BACKCHANNEL"], "明示的に待ってと言っているが、長すぎる沈黙では短い相づちも許容"
        return "WAIT", ["WAIT"], "話者が待つよう依頼している"
    if profile == "finished":
        return "SUPPORT", ["SUPPORT"], "発話が完了しており、短い沈黙でも応答してよい"
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


def natural_time(seconds: float, variant: int) -> str:
    short = {
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
    return short[seconds][variant % 3]


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
    if index % 7 != 0:
        return False, None
    values = [0.5, 1.0, 2.0, 5.0, 8.0]
    units = ["kg", "m", "点", "円"]
    value = values[(index // 7) % len(values)]
    unit = units[(index // 11) % len(units)]
    return True, f"[{value:g}{unit}]"


def make_fragment(profile: str, serial: int, used: set[str]) -> str:
    topic = TOPICS[(serial * 7 + len(profile)) % len(TOPICS)]
    template = TEMPLATES[profile][(serial * 5 + len(profile)) % len(TEMPLATES[profile])]
    fragment = template.format(topic=topic)
    if fragment in used:
        fragment = f"{fragment}（別の場面として）"
    if fragment in used:
        fragment = f"{fragment} {serial}"
    used.add(fragment)
    return fragment


def build_split(split: str, size: int, used_fragments: set[str], rng: random.Random) -> list[dict]:
    times = SPLIT_TIMES[split]
    candidate_pairs = []
    for profile in PROFILES:
        for seconds in times:
            label, acceptable, reason = label_for(profile, seconds)
            candidate_pairs.append((label, profile, seconds, acceptable, reason))
    by_label = defaultdict(list)
    for row in candidate_pairs:
        by_label[row[0]].append(row)
    for rows in by_label.values():
        rng.shuffle(rows)

    label_cycle = LABELS * ((size // len(LABELS)) + 2)
    examples = []
    counters = Counter()
    global_serial = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    for i in range(size):
        target_label = label_cycle[i]
        pool = by_label[target_label]
        label, profile, seconds, acceptable, reason = pool[counters[target_label] % len(pool)]
        counters[target_label] += 1
        fragment = make_fragment(profile, global_serial + i, used_fragments)
        cue, cue_type = time_expression(seconds, global_serial + i)
        has_control, control_note = negative_control(global_serial + i)
        examples.append(
            {
                "id": f"{split}_{i:04d}",
                "split": split,
                "profile": profile,
                "fragment": fragment,
                "seconds": seconds,
                "time_expression": cue,
                "time_expression_type": cue_type,
                "label": label,
                "acceptable_labels": acceptable,
                "rationale": reason,
                "has_negative_control": has_control,
                "unrelated_numeric_note": control_note,
            }
        )
    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    used_fragments: set[str] = set()
    all_rows = {}
    for split, size in SPLIT_SIZES.items():
        rows = build_split(split, size, used_fragments, rng)
        all_rows[split] = rows
        write_jsonl(OUT_DIR / f"{split}.jsonl", rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "labels": LABELS,
        "split_sizes": {split: len(rows) for split, rows in all_rows.items()},
        "split_times": SPLIT_TIMES,
        "profiles": PROFILES,
        "label_counts": {split: dict(Counter(row["label"] for row in rows)) for split, rows in all_rows.items()},
        "profile_counts": {split: dict(Counter(row["profile"] for row in rows)) for split, rows in all_rows.items()},
        "time_expression_counts": {
            split: dict(Counter(row["time_expression_type"] for row in rows)) for split, rows in all_rows.items()
        },
        "negative_control_counts": {
            split: sum(1 for row in rows if row["has_negative_control"]) for split, rows in all_rows.items()
        },
        "utterance_overlap": {
            "train_validation": len({r["fragment"] for r in all_rows["train"]} & {r["fragment"] for r in all_rows["validation"]}),
            "train_test": len({r["fragment"] for r in all_rows["train"]} & {r["fragment"] for r in all_rows["test"]}),
            "validation_test": len({r["fragment"] for r in all_rows["validation"]} & {r["fragment"] for r in all_rows["test"]}),
        },
        "notes": [
            "Labels are context-dependent; the same seconds can map to different labels across profiles.",
            "Validation and test include shifted/held-out time values such as 0.8, 1.5, 3.0, 6.0, and 7.0 seconds.",
            "Some examples include unrelated numeric notes as negative controls; these notes must not determine the label.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {OUT_DIR}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
