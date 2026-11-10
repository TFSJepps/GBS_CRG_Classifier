import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HIGH = 0.70    # confidence at or above this is "high"
LOW = 0.50     # below this is "low"
CLOSE = 0.15   # top-1 minus top-2 probability below this is a "close call"
REVIEW_ROWS = 25


def pct(x):
    return f"{x:.0%}"


def load(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        categories = next(reader)[5:]
        for r in reader:
            probs = [float(p) for p in r[5:]]
            order = sorted(range(len(probs)), key=lambda i: -probs[i])
            second = order[1] if len(order) > 1 else None
            rows.append({
                "rel": r[1], "cat": r[2], "conf": float(r[3]),
                "second": categories[second] if second is not None else None,
                "margin": probs[order[0]] - probs[second] if second is not None else 1.0,
            })
    return rows


def needs_review(r):
    return r["conf"] < LOW or r["margin"] < CLOSE


def folder_key(rows):
    # Shallowest folder level that actually splits the images (skips single wrapper folders).
    split = [r["rel"].replace("\\", "/").split("/") for r in rows]
    depth = 1
    while depth < 4 and len({tuple(p[:depth]) for p in split if len(p) > depth}) <= 1 and any(len(p) > depth + 1 for p in split):
        depth += 1
    return [("/".join(p[:depth]) if len(p) > depth else "(top level)") for p in split]


def write_summary(csv_path, *dest_dirs):
    """Write a human-readable JSON summary (<csv stem>_summary.txt) next to the CSV, and a copy into each dest_dir."""
    csv_path = Path(csv_path)
    rows = load(csv_path)
    if not rows:
        return None
    total = len(rows)
    high = sum(r["conf"] >= HIGH for r in rows)
    low = sum(r["conf"] < LOW for r in rows)
    review = [r for r in rows if needs_review(r)]

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
    folders = defaultdict(list)
    for r, key in zip(rows, folder_key(rows)):
        folders[key].append(r)
    pairs = Counter((r["cat"], r["second"]) for r in rows if r["margin"] < CLOSE and r["second"])

    summary = {
        "total_images": total,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_csv": csv_path.name,
        "overview": {
            "categories_found": [cat for cat, _ in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))],
            "images_worth_a_second_look": f"{len(review)} ({pct(len(review) / total)})",
        },
        "confidence_breakdown": {
            f"high (>= {pct(HIGH)})": high,
            f"medium ({pct(LOW)} to {pct(HIGH)})": total - high - low,
            f"low (< {pct(LOW)})": low,
        },
        "categories": {
            cat: {
                "images": len(rs),
                "share": pct(len(rs) / total),
                "avg_confidence": pct(sum(r["conf"] for r in rs) / len(rs)),
                "worth_a_second_look": sum(needs_review(r) for r in rs),
            }
            for cat, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))
        },
        "most_confused_pairs": [
            {"chosen": a, "nearly_chosen_instead": b, "images": n} for (a, b), n in pairs.most_common(10)
        ],
        "by_folder": {
            name: {
                "images": len(rs),
                "most_common_category": f"{Counter(r['cat'] for r in rs).most_common(1)[0][0]} "
                                        f"({pct(Counter(r['cat'] for r in rs).most_common(1)[0][1] / len(rs))})",
                "worth_a_second_look": sum(needs_review(r) for r in rs),
            }
            for name, rs in sorted(folders.items(), key=lambda kv: -len(kv[1]))[:30]
        },
        "review_these_first": [
            {"image": r["rel"], "chosen": r["cat"], "confidence": pct(r["conf"]), "runner_up": r["second"]}
            for r in sorted(review, key=lambda r: (r["conf"], r["margin"]))[:REVIEW_ROWS]
        ],
        "how_to_read": (
            "confidence = how strongly the model prefers its pick over your other categories (always sums to 100% across them, "
            "so an image that fits none can still score high). 'worth a second look' = confidence under "
            f"{pct(LOW)} or the top two categories within {pct(CLOSE)} of each other."
        ),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    path = csv_path.with_name(csv_path.stem + "_summary.txt")
    path.write_text(text, encoding="utf-8")
    for d in dest_dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "summary.txt").write_text(text, encoding="utf-8")
    return path
