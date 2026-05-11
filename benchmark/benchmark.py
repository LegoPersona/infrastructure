"""
Lego Model Benchmark 2.0
========================
Scores how well a Lego figure represents the original person photo.

Three components:
  1. Attribute match (60%) — CLIP zero-shot classification of 7 appearance
     features using domain-specific prompts: person prompts for the original
     photo, Lego-specific prompts for the generated figure. Each feature is
     scored with a dot-product over the probability distributions and weighted
     by its visual importance in a Lego figure.

  2. Visual similarity (20%) — CLIP ViT-L/14 cosine similarity, rescaled to
     remove the random-pair floor (~0.30 for this model) so unrelated images
     score near 0 instead of 0.5.

  3. Color match (20%) — H and S channel Bhattacharyya coefficients averaged
     independently so a saturation match alone cannot inflate the score.

Attribute categories mirror the lego-service template options:
  hair_color, hair_style, eye_color, eyebrow_color, eyebrow_shape,
  facial_hair, nose_shape

Folder structure expected:
    test_images/
        _ref_alice/         <- reference / golden example
            original.jpg
            lego.jpg
        person1/
            original.jpg
            lego.jpg

Usage:
    pip install -r requirements.txt
    python benchmark.py [--folder ./test_images] [--verbose]
"""

import argparse
import email
import json
import warnings
from pathlib import Path

import clip
import requests
import torch
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

_CLIP_FLOOR = 0.30  # empirical cosine-similarity floor for ViT-L/14 on random pairs

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess, device


def get_embedding(model, preprocess, device, image_path):
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def visual_similarity(emb_a, emb_b):
    raw = (emb_a * emb_b).sum().item()
    return max(0.0, (raw - _CLIP_FLOOR) / (1.0 - _CLIP_FLOOR))


def _clip_probs(model, device, image_emb, prompts):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    logits = (image_emb @ text_embs.T).squeeze(0) * 100
    return torch.softmax(logits, dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Attribute classification
# Separate prompt sets for each domain so CLIP classifies each image in its
# own visual language rather than forcing cross-domain text matching.
# Options mirror the lego-service template categories exactly.
# ---------------------------------------------------------------------------

# How much each attribute contributes to the final attribute score.
# Weights reflect visual prominence in a Lego figure.
ATTR_WEIGHTS = {
    "hair_color":    0.25,
    "hair_style":    0.20,
    "facial_hair":   0.20,
    "eye_color":     0.15,
    "eyebrow_color": 0.10,
    "eyebrow_shape": 0.05,
    "nose_shape":    0.05,
}

PERSON_PROMPTS = {
    "hair_color": [
        "a photo of a person with blonde or light hair",
        "a photo of a person with brown hair",
        "a photo of a person with black or dark hair",
        "a photo of a person with red or auburn hair",
        "a photo of a person with white or gray hair",
    ],
    "hair_style": [
        "a photo of a person with an afro or very curly hair",
        "a photo of a person with short cropped or crew cut hair",
        "a photo of a person with long wavy hair",
        "a photo of a person with medium length straight hair",
        "a photo of a person with a short bob hairstyle",
        "a photo of a person with short textured or messy hair",
        "a photo of a bald person",
    ],
    "eye_color": [
        "a photo of a person with black or very dark eyes",
        "a photo of a person with blue eyes",
        "a photo of a person with brown eyes",
        "a photo of a person with green eyes",
    ],
    "eyebrow_color": [
        "a photo of a person with black eyebrows",
        "a photo of a person with blonde or light eyebrows",
        "a photo of a person with brown eyebrows",
        "a photo of a person with gray eyebrows",
    ],
    "eyebrow_shape": [
        "a photo of a person with curved or arched eyebrows",
        "a photo of a person with round eyebrows",
        "a photo of a person with straight flat eyebrows",
        "a photo of a person with thick bushy eyebrows",
    ],
    "facial_hair": [
        "a photo of a person with a full beard",
        "a photo of a person with a goatee",
        "a photo of a person with no beard or facial hair",
    ],
    "nose_shape": [
        "a photo of a person with a long narrow nose",
        "a photo of a person with a pointed or sharp nose",
        "a photo of a person with a round or button nose",
    ],
}

LEGO_PROMPTS = {
    "hair_color": [
        "a lego minifigure with blonde or yellow hair",
        "a lego minifigure with brown hair",
        "a lego minifigure with black hair",
        "a lego minifigure with red or orange hair",
        "a lego minifigure with white or gray hair",
    ],
    "hair_style": [
        "a lego minifigure with an afro hairstyle",
        "a lego minifigure with short cropped hair",
        "a lego minifigure with long wavy hair",
        "a lego minifigure with medium length straight hair",
        "a lego minifigure with a short bob hairstyle",
        "a lego minifigure with short textured hair",
        "a lego minifigure with no hair piece or bald head",
    ],
    "eye_color": [
        "a lego minifigure with black or dark eyes",
        "a lego minifigure with blue eyes",
        "a lego minifigure with brown eyes",
        "a lego minifigure with green eyes",
    ],
    "eyebrow_color": [
        "a lego minifigure with black eyebrows",
        "a lego minifigure with blonde or light eyebrows",
        "a lego minifigure with brown eyebrows",
        "a lego minifigure with gray eyebrows",
    ],
    "eyebrow_shape": [
        "a lego minifigure with curved or arched eyebrows",
        "a lego minifigure with round eyebrows",
        "a lego minifigure with straight eyebrows",
        "a lego minifigure with thick eyebrows",
    ],
    "facial_hair": [
        "a lego minifigure with a full beard",
        "a lego minifigure with a goatee",
        "a lego minifigure with no beard",
    ],
    "nose_shape": [
        "a lego minifigure with a long nose",
        "a lego minifigure with a pointed nose",
        "a lego minifigure with a round nose",
    ],
}


def attribute_scores(model, device, orig_emb, lego_emb):
    """Returns per-attribute dot-product scores (0–1 each)."""
    scores = {}
    for attr, weight in ATTR_WEIGHTS.items():
        p_orig = _clip_probs(model, device, orig_emb, PERSON_PROMPTS[attr])
        p_lego = _clip_probs(model, device, lego_emb, LEGO_PROMPTS[attr])
        scores[attr] = float(np.dot(p_orig, p_lego))
    return scores


def weighted_attr_score(scores):
    return sum(ATTR_WEIGHTS[attr] * s for attr, s in scores.items())


# ---------------------------------------------------------------------------
# Color similarity
# ---------------------------------------------------------------------------

def _hsv_histograms(image_path, bins=32):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    s = np.where(maxc != 0, (maxc - minc) / maxc, 0)
    diff = np.where(maxc - minc == 0, 1, maxc - minc)
    h = np.zeros_like(r)
    m = maxc == r; h[m] = (60 * ((g[m] - b[m]) / diff[m])) % 360
    m = maxc == g; h[m] = 60 * ((b[m] - r[m]) / diff[m]) + 120
    m = maxc == b; h[m] = 60 * ((r[m] - g[m]) / diff[m]) + 240
    h /= 360.0
    h_hist, _ = np.histogram(h.flatten(), bins=bins, range=(0, 1))
    s_hist, _ = np.histogram(s.flatten(), bins=bins, range=(0, 1))
    h_hist = h_hist.astype(np.float32) / h_hist.sum()
    s_hist = s_hist.astype(np.float32) / s_hist.sum()
    return h_hist, s_hist


def color_similarity(path_a, path_b):
    """
    Bhattacharyya coefficient computed per channel (H and S) then averaged.
    Each channel is independently bounded [0, 1] — a saturation match alone
    cannot push the total to 1.0 when the hues are completely different.
    """
    h_a, s_a = _hsv_histograms(path_a)
    h_b, s_b = _hsv_histograms(path_b)
    score_h = float(np.sum(np.sqrt(h_a * h_b)))
    score_s = float(np.sum(np.sqrt(s_a * s_b)))
    return (score_h + score_s) / 2


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

VISUAL_WEIGHT = 0.20
ATTR_WEIGHT   = 0.60
COLOR_WEIGHT  = 0.20


def combined_score(visual, attr, color):
    return VISUAL_WEIGHT * visual + ATTR_WEIGHT * attr + COLOR_WEIGHT * color


# ---------------------------------------------------------------------------
# Per-pair scoring
# ---------------------------------------------------------------------------

def score_pair(original_path, lego_path, model, preprocess, device):
    orig_emb = get_embedding(model, preprocess, device, original_path)
    lego_emb = get_embedding(model, preprocess, device, lego_path)

    vis   = visual_similarity(orig_emb, lego_emb)
    attrs = attribute_scores(model, device, orig_emb, lego_emb)
    attr  = weighted_attr_score(attrs)
    color = color_similarity(original_path, lego_path)
    final = combined_score(vis, attr, color)
    return vis, attr, attrs, color, final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def relative_score(score, baseline):
    return 0.0 if baseline == 0 else min(1.0, score / baseline)


def rating(pct):
    if pct >= 0.90: return "✅ Good"
    if pct >= 0.70: return "⚠️  OK"
    return "❌ Poor"


def collect_pairs(root, prefix):
    pairs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        is_ref = d.name.startswith("_ref")
        if prefix == "ref" and not is_ref: continue
        if prefix == "test" and is_ref:    continue
        orig = next((d / f for f in ("original.jpg", "original.png") if (d / f).exists()), None)
        lego = next((d / f for f in ("lego.jpg", "lego.png")         if (d / f).exists()), None)
        if not orig or not lego:
            print(f"  [skip] {d.name} — missing original or lego image")
            continue
        pairs.append((d.name, orig, lego))
    return pairs


def print_attr_breakdown(attrs):
    parts = "  ".join(f"{k}: {v:.2f}" for k, v in attrs.items())
    print(f"    {parts}")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate_lego_images(folder, api_url):
    root = Path(folder)
    test_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_ref")]
    if not test_dirs:
        print("[generate] No test folders found.")
        return
    for person_dir in test_dirs:
        orig = next((person_dir / f for f in ("original.jpg", "original.png") if (person_dir / f).exists()), None)
        if not orig:
            print(f"[generate] {person_dir.name} — no original image found, skipping")
            continue
        try:
            mime = "image/jpeg" if orig.suffix.lower() == ".jpg" else "image/png"
            with open(orig, "rb") as f:
                resp = requests.post(f"{api_url}/api/v1/personas",
                                     files={"image": (orig.name, f, mime)}, timeout=120)
            resp.raise_for_status()
            ct  = resp.headers.get("Content-Type", "")
            msg = email.message_from_bytes(b"Content-Type: " + ct.encode() + b"\r\n\r\n" + resp.content)
            pid = None
            for part in msg.walk():
                if part.get_content_type() == "application/json":
                    pid = json.loads(part.get_payload(decode=True)).get("id")
                    break
            if not pid:
                print(f"[generate] {person_dir.name} — could not extract persona id, skipping")
                continue
            img_resp = requests.get(f"{api_url}/api/v1/personas/{pid}/image", timeout=120)
            img_resp.raise_for_status()
            (person_dir / "lego.png").write_bytes(img_resp.content)
            print(f"[generate] {person_dir.name} → lego.png")
        except Exception as exc:
            print(f"[generate] {person_dir.name} — error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(folder, verbose=False):
    root = Path(folder)
    ref_pairs  = collect_pairs(root, "ref")
    test_pairs = collect_pairs(root, "test")

    if not ref_pairs:
        print("No reference folders found (folders starting with _ref).")
        return
    if not test_pairs:
        print("No test pairs found.")
        return

    print("Loading CLIP model (ViT-L/14)...")
    model, preprocess, device = load_clip_model()
    print(f"Running on: {device}\n")

    col_n   = 22
    header  = f"{'Name':<{col_n}} {'Visual':>7} {'Attr':>7} {'Color':>7} {'Score':>7} {'Relative':>9}  Rating"
    divider = "-" * (len(header) + 2)

    # --- References ---
    print("=== Reference Pairs ===")
    print(header)
    print(divider)

    ref_scores = []
    for name, orig, lego in ref_pairs:
        vis, attr, attrs, color, final = score_pair(orig, lego, model, preprocess, device)
        ref_scores.append(final)
        print(f"{name:<{col_n}} {vis:>7.4f} {attr:>7.4f} {color:>7.4f} {final:>7.4f} {'(baseline)':>9}")
        if verbose:
            print_attr_breakdown(attrs)

    baseline = sum(ref_scores) / len(ref_scores)
    print(divider)
    print(f"  Baseline (avg of {len(ref_scores)} reference(s)): {baseline:.4f}\n")

    # --- Test pairs ---
    print("=== Test Pairs ===")
    print(header)
    print(divider)

    relatives = []
    for name, orig, lego in test_pairs:
        vis, attr, attrs, color, final = score_pair(orig, lego, model, preprocess, device)
        rel = relative_score(final, baseline)
        relatives.append(rel)
        print(f"{name:<{col_n}} {vis:>7.4f} {attr:>7.4f} {color:>7.4f} {final:>7.4f} {rel:>9.1%}  {rating(rel)}")
        if verbose:
            print_attr_breakdown(attrs)

    print(divider)
    avg_rel = sum(relatives) / len(relatives)
    print(f"\nTotal test pairs:  {len(test_pairs)}")
    print(f"Baseline score:    {baseline:.4f}")
    print(f"Average relative:  {avg_rel:.1%}")
    print(f"Weights: Visual {int(VISUAL_WEIGHT*100)}% / Attr {int(ATTR_WEIGHT*100)}% / Color {int(COLOR_WEIGHT*100)}%")

    if avg_rel >= 0.90:   print("\n🟢 Pipeline result: STRONG")
    elif avg_rel >= 0.70: print("\n🟡 Pipeline result: ACCEPTABLE")
    else:                 print("\n🔴 Pipeline result: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lego persona benchmark 2.0")
    parser.add_argument("--folder",   default="./test_images",    help="Path to test images folder")
    parser.add_argument("--generate", action="store_true",         help="Generate lego images via the API before benchmarking")
    parser.add_argument("--api-url",  default="http://localhost:3000", help="Backend API base URL")
    parser.add_argument("--verbose",  action="store_true",         help="Print per-attribute score breakdown for each pair")
    args = parser.parse_args()
    if args.generate:
        generate_lego_images(args.folder, args.api_url)
    run_benchmark(args.folder, verbose=args.verbose)
