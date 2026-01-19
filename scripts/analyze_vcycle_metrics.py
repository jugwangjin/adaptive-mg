#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

TAG_ORDER = [
    "DownPreSmooth",
    "Down",
    "DownPostRestrict",
    "UpPreProlong",
    "UpPreSmooth",
    "UpPostSmooth",
]

METRICS = ["psnr", "ssim", "lpips"]


def tag_prefix(tag: str) -> str:
    if not tag:
        return ""
    parts = tag.rsplit(" ", 1)
    return parts[0] if len(parts) > 1 else tag


def load_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(values):
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def analyze(records, stage: str | None):
    transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    net = defaultdict(lambda: defaultdict(list))

    state = {}

    for rec in records:
        if stage and rec.get("stage") != stage:
            continue

        level = rec.get("level")
        prefix = tag_prefix(rec.get("tag", ""))
        if prefix not in TAG_ORDER:
            continue

        if level not in state:
            state[level] = {
                "last_prefix": None,
                "last_rec": None,
                "current": {},
            }

        st = state[level]

        if prefix == "DownPreSmooth":
            st["current"] = {"DownPreSmooth": rec}
            st["last_prefix"] = prefix
            st["last_rec"] = rec
            continue

        if st["last_prefix"] is None:
            # Wait for DownPreSmooth to start a clean sequence.
            continue

        prev_idx = TAG_ORDER.index(st["last_prefix"])
        curr_idx = TAG_ORDER.index(prefix)

        # Only compute deltas when tags come in expected order.
        if curr_idx == prev_idx + 1 and st["last_rec"] is not None:
            transition = f"{st['last_prefix']}->{prefix}"
            for m in METRICS:
                transitions[level][transition][m].append(rec[m] - st["last_rec"][m])

        st["last_prefix"] = prefix
        st["last_rec"] = rec
        st["current"][prefix] = rec

        if prefix == "UpPostSmooth" and "DownPreSmooth" in st["current"]:
            base = st["current"]["DownPreSmooth"]
            for m in METRICS:
                net[level][m].append(rec[m] - base[m])

    return transitions, net


def main():
    parser = argparse.ArgumentParser(description="Analyze V-cycle metric deltas from jsonl log.")
    parser.add_argument("path", help="Path to vcycle_metrics.jsonl")
    parser.add_argument("--stage", default="val", help="Stage filter (default: val)")
    args = parser.parse_args()

    records = load_records(args.path)
    transitions, net = analyze(records, args.stage if args.stage else None)

    print(f"Records: {len(records)}")
    print(f"Stage: {args.stage}")
    print()

    for level in sorted(transitions.keys()):
        print(f"Level {level}:")
        for i in range(1, len(TAG_ORDER)):
            prev_tag = TAG_ORDER[i - 1]
            curr_tag = TAG_ORDER[i]
            key = f"{prev_tag}->{curr_tag}"
            if key not in transitions[level]:
                continue
            print(f"  {key}")
            for m in METRICS:
                s = summarize(transitions[level][key][m])
                if not s:
                    continue
                print(
                    f"    {m}: mean {s['mean']:+.3f}, min {s['min']:+.3f}, "
                    f"max {s['max']:+.3f}, n={s['n']}"
                )
        if level in net:
            print("  DownPreSmooth->UpPostSmooth")
            for m in METRICS:
                s = summarize(net[level][m])
                if not s:
                    continue
                print(
                    f"    {m}: mean {s['mean']:+.3f}, min {s['min']:+.3f}, "
                    f"max {s['max']:+.3f}, n={s['n']}"
                )
        print()


if __name__ == "__main__":
    main()

