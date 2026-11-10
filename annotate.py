import json

from PIL import Image, ImageDraw, ImageFont


def make_preview(im, max_side):
    preview = im.copy()
    preview.thumbnail((max_side, max_side))
    return preview


def write_annotation(out_dir, image, preview, categories, probs, top, match):
    """Write <rel>.jpg (preview with a label bar) and <rel>.txt (JSON scores) under out_dir, mirroring the source tree."""
    dest = out_dir / image.rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = f"{categories[top]}  {probs[top]:.1%}"

    w, h = preview.size
    size = max(14, w // 24)
    font = ImageFont.load_default(size=size)
    while size > 10 and ImageDraw.Draw(preview).textlength(label, font=font) > w - 16:
        size -= 2
        font = ImageFont.load_default(size=size)
    bar = size + 16
    canvas = Image.new("RGB", (w, h + bar), (0, 0, 0))
    canvas.paste(preview, (0, 0))
    ImageDraw.Draw(canvas).text((8, h + 8), label, font=font, fill=(255, 255, 255))
    canvas.save(dest.with_name(dest.name + ".jpg"), quality=85)

    record = {
        "image": image.rel,
        "source": image.ref,
        "predicted_category": categories[top],
        "confidence": round(probs[top], 4),
        "match_score": None if match is None else round(match, 4),
        "scores": {c: round(p, 4) for c, p in sorted(zip(categories, probs), key=lambda cp: -cp[1])},
    }
    dest.with_name(dest.name + ".txt").write_text(json.dumps(record, indent=2), encoding="utf-8")
