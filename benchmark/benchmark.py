"""
Lego Model Benchmark — CLIP + Face + Color Similarity Score
============================================================
Measures how similar each Lego output is to its original photo,
using a combination of:
  - CLIP (semantic similarity)
  - DeepFace (face identity)
  - Color histogram (skin/hair color match)

Reference folders (prefixed with _ref) establish a baseline score.
All other folders are scored relative to that baseline.

Folder structure expected:
    test_images/
        _ref_alice/         <- reference / golden example
            original.jpg
            lego.jpg
        person1/
            original.jpg
            lego.jpg
        ...

Usage:
    pip install -r requirements.txt
    python benchmark.py --folder ./test_images
"""

import os

# Suppress TensorFlow / DeepFace console noise
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
import argparse
import email
import json
import requests
import warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
from pathlib import Path

import clip
import torch
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# --- CLIP -------------------------------------------------------------------

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def get_clip_embedding(model, preprocess, device, image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def clip_similarity(model, preprocess, device, path_a, path_b):
    emb_a = get_clip_embedding(model, preprocess, device, path_a)
    emb_b = get_clip_embedding(model, preprocess, device, path_b)
    return (emb_a * emb_b).sum().item()


# --- DeepFace ---------------------------------------------------------------

def face_similarity(path_a, path_b):
    try:
        from deepface import DeepFace
        result = DeepFace.verify(
            img1_path=str(path_a),
            img2_path=str(path_b),
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        return max(0.0, 1.0 - result["distance"])
    except Exception:
        return None


# --- Color similarity -------------------------------------------------------

def get_color_histogram(image_path, bins=32):
    """
    Returns a normalized HSV color histogram for the image.
    Using HSV because hue captures hair/skin color better than RGB.
    We focus on the H and S channels and ignore V (brightness),
    so lighting differences don't throw off the comparison.
    """
    img = Image.open(image_path).convert("RGB")
    # Resize for speed
    img = img.resize((128, 128))
    img_hsv = img.convert("HSV") if hasattr(img, "convert") else img
    # PIL doesn't support HSV natively — use numpy
    arr = np.array(img).astype(np.float32) / 255.0
    # Convert RGB to HSV manually
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    v = maxc
    s = np.where(maxc != 0, (maxc - minc) / maxc, 0)
    diff = maxc - minc
    diff = np.where(diff == 0, 1, diff)  # avoid division by zero
    h = np.zeros_like(r)
    mask = maxc == r
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask])) % 360
    mask = maxc == g
    h[mask] = 60 * ((b[mask] - r[mask]) / diff[mask]) + 120
    mask = maxc == b
    h[mask] = 60 * ((r[mask] - g[mask]) / diff[mask]) + 240
    h = h / 360.0  # normalize to 0-1

    # Build histogram on H and S (ignore V/brightness)
    h_hist, _ = np.histogram(h.flatten(), bins=bins, range=(0, 1))
    s_hist, _ = np.histogram(s.flatten(), bins=bins, range=(0, 1))

    # Normalize
    h_hist = h_hist.astype(np.float32) / h_hist.sum()
    s_hist = s_hist.astype(np.float32) / s_hist.sum()

    return np.concatenate([h_hist, s_hist])


def color_similarity(path_a, path_b):
    """
    Compares color histograms using the Bhattacharyya coefficient,
    which measures overlap between two distributions. Returns 0-1.
    """
    hist_a = get_color_histogram(path_a)
    hist_b = get_color_histogram(path_b)
    # Bhattacharyya coefficient
    score = float(np.sum(np.sqrt(hist_a * hist_b)))
    return min(1.0, score)


# --- Combined scoring -------------------------------------------------------

CLIP_WEIGHT  = 0.25
FACE_WEIGHT  = 0.50
COLOR_WEIGHT = 0.25


def combined_score(clip_score, face_score, color_score):
    if face_score is None:
        # No face: split between CLIP and color
        score = 0.5 * clip_score + 0.5 * color_score
        return score, "CLIP+color"
    score = CLIP_WEIGHT * clip_score + FACE_WEIGHT * face_score + COLOR_WEIGHT * color_score
    return score, "combined"


# --- Helpers ----------------------------------------------------------------

def score_pair(name, original_path, lego_path, clip_model, preprocess, device):
    c_score = clip_similarity(clip_model, preprocess, device, original_path, lego_path)
    f_score = face_similarity(original_path, lego_path)
    col_score = color_similarity(original_path, lego_path)
    final, mode = combined_score(c_score, f_score, col_score)
    return c_score, f_score, col_score, final, mode


def relative_score(score, baseline):
    if baseline == 0:
        return 0.0
    return min(1.0, score / baseline)


def rating(pct):
    if pct >= 0.90:
        return "✅ Good"
    elif pct >= 0.70:
        return "⚠️  OK"
    else:
        return "❌ Poor"


def collect_pairs(root, prefix):
    pairs = []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        is_ref = person_dir.name.startswith("_ref")
        if prefix == "ref" and not is_ref:
            continue
        if prefix == "test" and is_ref:
            continue
        original = person_dir / "original.jpg"
        lego = person_dir / "lego.jpg"
        if not original.exists():
            original = person_dir / "original.png"
        if not lego.exists():
            lego = person_dir / "lego.png"
        if not original.exists() or not lego.exists():
            print(f"  [skip] {person_dir.name} — missing original or lego image")
            continue
        pairs.append((person_dir.name, original, lego))
    return pairs


# --- Generate ---------------------------------------------------------------

def generate_lego_images(folder: str, api_url: str) -> None:
    root = Path(folder)
    test_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_ref")]

    if not test_dirs:
        print("[generate] No test folders found.")
        return

    for person_dir in test_dirs:
        original = person_dir / "original.jpg"
        if not original.exists():
            original = person_dir / "original.png"
        if not original.exists():
            print(f"[generate] {person_dir.name} — no original image found, skipping")
            continue

        try:
            mime_type = "image/jpeg" if original.suffix.lower() == ".jpg" else "image/png"
            with open(original, "rb") as f:
                create_resp = requests.post(
                    f"{api_url}/api/v1/personas",
                    files={"image": (original.name, f, mime_type)},
                    timeout=120,
                )
            create_resp.raise_for_status()

            content_type = create_resp.headers.get("Content-Type", "")
            raw = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + create_resp.content
            msg = email.message_from_bytes(raw)
            persona_id = None
            for part in msg.walk():
                if part.get_content_type() == "application/json":
                    data = json.loads(part.get_payload(decode=True))
                    persona_id = data.get("id")
                    break

            if not persona_id:
                print(f"[generate] {person_dir.name} — could not extract persona id, skipping")
                continue

            img_resp = requests.get(
                f"{api_url}/api/v1/personas/{persona_id}/image",
                timeout=120,
            )
            img_resp.raise_for_status()

            lego_path = person_dir / "lego.png"
            lego_path.write_bytes(img_resp.content)
            print(f"[generate] {person_dir.name} → lego.png")

        except Exception as exc:
            print(f"[generate] {person_dir.name} — error: {exc}")


# --- Main -------------------------------------------------------------------

def run_benchmark(folder: str):
    root = Path(folder)
    ref_pairs  = collect_pairs(root, "ref")
    test_pairs = collect_pairs(root, "test")

    if not ref_pairs:
        print("No reference folders found (folders starting with _ref).")
        return
    if not test_pairs:
        print("No test pairs found.")
        return

    print("Loading CLIP model...")
    clip_model, preprocess, device = load_clip_model()
    print(f"Running on: {device}")
    print("Loading DeepFace...\n")

    col_n  = 22
    header = (
        f"{'Name':<{col_n}} {'CLIP':>7} {'Face':>7} {'Color':>7} "
        f"{'Score':>7} {'Relative':>9}  Rating"
    )
    divider = "-" * (len(header) + 2)

    # --- References ---
    print("=== Reference Pairs ===")
    print(header)
    print(divider)

    ref_scores = []
    for name, orig, lego in ref_pairs:
        c, f, col, final, mode = score_pair(name, orig, lego, clip_model, preprocess, device)
        ref_scores.append(final)
        face_str = f"{f:.4f}" if f is not None else "  n/a "
        print(
            f"{name:<{col_n}} {c:>7.4f} {face_str:>7} {col:>7.4f} "
            f"{final:>7.4f} {'(baseline)':>9}  ({mode})"
        )

    baseline = sum(ref_scores) / len(ref_scores)
    print(divider)
    print(f"  Baseline (avg of {len(ref_scores)} reference(s)): {baseline:.4f}\n")

    # --- Test pairs ---
    print("=== Test Pairs ===")
    print(header)
    print(divider)

    relatives = []
    for name, orig, lego in test_pairs:
        c, f, col, final, mode = score_pair(name, orig, lego, clip_model, preprocess, device)
        rel = relative_score(final, baseline)
        relatives.append(rel)
        face_str = f"{f:.4f}" if f is not None else "  n/a "
        print(
            f"{name:<{col_n}} {c:>7.4f} {face_str:>7} {col:>7.4f} "
            f"{final:>7.4f} {rel:>9.1%}  {rating(rel)}  ({mode})"
        )

    print(divider)
    avg_rel = sum(relatives) / len(relatives)
    print(f"\nTotal test pairs:  {len(test_pairs)}")
    print(f"Baseline score:    {baseline:.4f}")
    print(f"Average relative:  {avg_rel:.1%}")
    print(f"Weights: CLIP {int(CLIP_WEIGHT*100)}% / Face {int(FACE_WEIGHT*100)}% / Color {int(COLOR_WEIGHT*100)}%")

    if avg_rel >= 0.90:
        print("\n🟢 Pipeline result: STRONG")
    elif avg_rel >= 0.70:
        print("\n🟡 Pipeline result: ACCEPTABLE")
    else:
        print("\n🔴 Pipeline result: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lego model benchmark")
    parser.add_argument(
        "--folder",
        type=str,
        default="./test_images",
        help="Path to test images folder (default: ./test_images)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate lego images via the backendAPI before benchmarking",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:3000",
        help="Base URL for the backendAPI (default: http://localhost:3000)",
    )
    args = parser.parse_args()
    if args.generate:
        generate_lego_images(args.folder, args.api_url)
    run_benchmark(args.folder)