import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from transformers.utils import logging as hf_logging

from annotate import make_preview, write_annotation
from sources import GraphSource, LocalSource
from summary import write_summary

# SigLIP 2 checkpoints ship a text config with CLIP's bos/eos ids, which triggers a harmless but alarming warning.
hf_logging.set_verbosity_error()

PROJECT_DIR = Path(__file__).resolve().parent
GPU_MODEL = "google/siglip2-so400m-patch14-384"
CPU_MODEL = "google/siglip2-base-patch16-224"


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot classify every image under a folder tree.")
    p.add_argument("--folder", type=Path, help="Root folder to scan (skips the folder picker).")
    p.add_argument("--sharepoint-url", help="Link to a SharePoint/OneDrive folder; images are streamed, not downloaded.")
    p.add_argument("--thumbnails", action="store_true", help="With --sharepoint-url, fetch 384px thumbnails instead of full files (much faster).")
    p.add_argument("--resume", action="store_true", help="Append to the latest (or --output) CSV, skipping images already in it.")
    p.add_argument("--workers", type=int, help="Loader threads (default: 8 local, 16 SharePoint).")
    p.add_argument("--annotate-dir", nargs="?", const="auto", type=Path, help="Also write a parallel folder of labelled preview images plus a JSON-style .txt per image and a summary.txt. Without a value: results/<csv name>_annotated.")
    p.add_argument("--annotate-size", type=int, default=1024, help="Longest side of the annotated previews in pixels (default 1024).")
    p.add_argument("--categories", type=Path, default=PROJECT_DIR / "categories.json")
    p.add_argument("--model", help="Hugging Face checkpoint id, SigLIP or CLIP family.")
    p.add_argument("--template", default="this is a photo of {}.", help="Prompt template; {} becomes each category.")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--output", type=Path, help="CSV path (default: results/classification_<timestamp>.csv).")
    return p.parse_args()


def load_categories(path):
    try:
        categories = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON (check for a trailing comma or missing quote): {e}")
    if not isinstance(categories, list) or not categories or not all(isinstance(c, str) for c in categories):
        sys.exit(f"{path} must be a non-empty JSON list of strings.")
    return categories


def pick_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select the root image folder to classify", mustexist=True)
    root.destroy()
    return Path(folder) if folder else None


def resolve_output(args):
    results = PROJECT_DIR / "results"
    if args.output:
        return args.output
    if args.resume:
        existing = sorted(results.glob("classification_*.csv"))
        if existing:
            return existing[-1]
    return results / f"classification_{datetime.now():%Y%m%d_%H%M%S}.csv"


def already_done(output):
    with open(output, newline="", encoding="utf-8-sig") as f:
        return {row["relative_path"] for row in csv.DictReader(f)}


class ZeroShotClassifier:
    def __init__(self, checkpoint, device):
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint, dtype=self.dtype).to(device).eval()
        # SigLIP is trained with a sigmoid loss and has a learned bias; CLIP uses softmax only.
        self.is_siglip = hasattr(self.model, "logit_bias")
        self.text_emb = None

    @torch.inference_mode()
    def set_labels(self, prompts):
        inputs = self.processor(text=prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        self.text_emb = F.normalize(self.model.get_text_features(**inputs.to(self.device)).pooler_output, dim=-1)

    def preprocess(self, image, preview_size=None):
        with image.load() as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        preview = make_preview(im, preview_size) if preview_size else None
        return self.processor(images=im, return_tensors="pt")["pixel_values"][0], preview

    @torch.inference_mode()
    def classify(self, pixels):
        out = self.model.get_image_features(pixel_values=pixels.to(self.device, self.dtype))
        img_emb = F.normalize(out.pooler_output, dim=-1)
        logits = img_emb @ self.text_emb.T * self.model.logit_scale.exp()
        if self.is_siglip:
            logits = logits + self.model.logit_bias
        logits = logits.float()
        match = logits.sigmoid().max(dim=-1).values if self.is_siglip else None
        return logits.softmax(dim=-1).cpu(), match.cpu() if match is not None else None


def safe_preprocess(clf, path, preview_size):
    try:
        return clf.preprocess(path, preview_size)
    except Exception as e:
        return e


def report_writes(futures):
    pending = []
    for fut in futures:
        if not fut.done():
            pending.append(fut)
        elif fut.exception():
            tqdm.write(f"Annotation failed: {fut.exception()}", file=sys.stderr)
    return pending


def main():
    args = parse_args()
    categories = load_categories(args.categories)

    if not args.sharepoint_url and not args.folder:
        args.sharepoint_url = input("Paste a SharePoint/OneDrive folder link (or press Enter for a local folder picker): ").strip().strip('"') or None

    if args.sharepoint_url:
        source = GraphSource(args.sharepoint_url, PROJECT_DIR / ".token_cache.json", args.thumbnails)
    else:
        folder = args.folder or pick_folder()
        if not folder:
            sys.exit("No folder selected.")
        if not folder.is_dir():
            sys.exit(f"Not a folder: {folder}")
        source = LocalSource(folder)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    checkpoint = args.model or (GPU_MODEL if cuda else CPU_MODEL)
    batch_size = args.batch_size or (32 if cuda else 8)
    print(f"Device: {torch.cuda.get_device_name(0) if cuda else 'CPU'}")

    print("Scanning ...")
    images = source.list_images()
    print(f"Found {len(images)} images")

    output = resolve_output(args)
    resuming = args.resume and output.exists()
    if resuming:
        done = already_done(output)
        images = [im for im in images if im.rel not in done]
        print(f"Resuming {output.name}: {len(done)} already done, {len(images)} to go")
    if not images:
        return

    print(f"Loading {checkpoint} ...")
    clf = ZeroShotClassifier(checkpoint, device)
    clf.set_labels([args.template.format(c).lower() for c in categories])

    output.parent.mkdir(parents=True, exist_ok=True)
    annotate_dir = None
    if args.annotate_dir:
        annotate_dir = output.with_name(output.stem + "_annotated") if str(args.annotate_dir) == "auto" else args.annotate_dir
        print(f"Annotated copies -> {annotate_dir}")
    preview_size = args.annotate_size if annotate_dir else None
    writes = []
    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
    counts = Counter()
    failed = 0

    # utf-8-sig so Excel detects the encoding when the report is double-clicked.
    workers = args.workers or (16 if args.sharepoint_url else 8)
    mode = "a" if resuming else "w"
    with open(output, mode, newline="", encoding="utf-8-sig" if not resuming else "utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        writer = csv.writer(f)
        if not resuming:
            writer.writerow(["image_path", "relative_path", "predicted_category", "confidence", "match_score", *categories])
        pending = [pool.submit(safe_preprocess, clf, p, preview_size) for p in batches[0]]

        with tqdm(total=len(images), unit="img") as bar:
            for i, batch in enumerate(batches):
                results = [fut.result() for fut in pending]
                # Prefetch the next batch while the GPU works on this one.
                if i + 1 < len(batches):
                    pending = [pool.submit(safe_preprocess, clf, p, preview_size) for p in batches[i + 1]]

                good = []
                for image, result in zip(batch, results):
                    if isinstance(result, Exception):
                        failed += 1
                        tqdm.write(f"Skipped {image.rel}: {result}", file=sys.stderr)
                    else:
                        good.append((image, result))

                if good:
                    probs, match = clf.classify(torch.stack([r[0] for _, r in good]))
                    for j, (image, (_, preview)) in enumerate(good):
                        top = int(probs[j].argmax())
                        if annotate_dir:
                            writes.append(pool.submit(
                                write_annotation, annotate_dir, image, preview, categories,
                                probs[j].tolist(), top, None if match is None else float(match[j]),
                            ))
                        counts[categories[top]] += 1
                        writer.writerow([
                            image.ref,
                            image.rel,
                            categories[top],
                            f"{probs[j, top]:.4f}",
                            f"{match[j]:.4f}" if match is not None else "",
                            *(f"{p:.4f}" for p in probs[j].tolist()),
                        ])
                    f.flush()
                writes = report_writes(writes)
                bar.update(len(batch))

        if annotate_dir:
            for fut in writes:
                fut.exception()  # wait for outstanding annotation writes
            report_writes(writes)

    print(f"\nResults written to {output}")
    summary = write_summary(output, annotate_dir)
    if summary:
        print(f"Readable summary: {summary}")
    if annotate_dir:
        print(f"Annotated images and per-image .txt files: {annotate_dir}")
    for category, n in counts.most_common():
        print(f"  {n:6d}  {category}")
    if failed:
        print(f"  {failed:6d}  (unreadable, skipped)")


if __name__ == "__main__":
    main()
