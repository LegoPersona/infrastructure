# Lego Model Benchmark

An automated tool that measures how well your Lego model creation pipeline preserves the likeness of the original person, scored **relative to your own golden examples**.

---

## How It Works

The benchmark uses **three signals** in combination:

| Model | What it measures | Weight |
|---|---|---|
| CLIP | Overall semantic similarity (style, concept, general appearance) | 25% |
| DeepFace | Face identity — is this the same person? | 50% |
| Color histogram | Do the dominant colors match? (hair, skin tone) | 25% |

Final score = `0.25 × CLIP + 0.50 × Face + 0.25 × Color`

### Why three signals?

- **CLIP alone** can't tell apart two different people rendered in the same Lego style — it just sees "person + Lego figure" and scores everything similarly.
- **DeepFace alone** struggles with stylized Lego images since it was trained on real faces.
- **Color** is the key differentiator for mismatches like a dark-skinned person with an afro being rendered as a blonde Lego figure — a pure color comparison catches this immediately.

Together they cover each other's blind spots.

### Reference-based scoring

Rather than fixed absolute thresholds, the benchmark calibrates against **your own golden examples** — pairs you've manually decided look good, stored in `_ref` folders.

1. Score all `_ref` folders → average them → **baseline**
2. Score all test folders → express each as a **% of baseline**
3. ≥90% = Good, 70–90% = OK, <70% = Poor

---

## What is CLIP?

CLIP (Contrastive Language-Image Pretraining) is a neural network by OpenAI trained on hundreds of millions of image-text pairs. It converts any image into a vector embedding encoding its semantic meaning. Images with similar meaning end up with similar vectors even if they look visually different — a photo and a Lego figure of the same person will score higher than a photo and a completely unrelated Lego figure.

---

## What is DeepFace?

DeepFace is a face recognition library using the **Facenet** model, trained to answer "are these two images of the same person?" It compares facial embeddings and returns a distance score. This is much more sensitive to person identity than CLIP, which is why it carries the most weight.

---

## What is Color Histogram Similarity?

Color histograms capture the distribution of colors across an image. This benchmark compares hue and saturation (ignoring brightness so lighting differences don't interfere) using the **Bhattacharyya coefficient** — a measure of overlap between two color distributions.

A blonde Lego figure generated for a dark-haired person will have a very different hue distribution and score low here, even if CLIP and DeepFace miss it.

---

## Folder Structure

```
benchmark/
    benchmark.py
    requirements.txt
    README.md
    test_images/
        _ref_alice/         ← golden reference example
            original.jpg
            lego.jpg
        _ref_bob/           ← another reference (optional)
            original.jpg
            lego.jpg
        person1/            ← test case
            original.jpg
            lego.jpg        ← generated automatically with --generate
        person2/
            original.jpg
            lego.jpg
```

- Any folder starting with `_ref` is a reference (used to build the baseline)
- Everything else is a test case scored against the baseline
- `.png` is supported alongside `.jpg`

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads CLIP weights (~350MB) and DeepFace/Facenet weights (~250MB). Both cached after that.

### 2. Run

```bash
python benchmark.py
```

Or with a custom folder:

```bash
python benchmark.py --folder ./my_test_set
```

---

## Auto-generating Lego images

If you only have `original.*` images and want to generate the lego images automatically using the live pipeline, use `--generate`:

```bash
python benchmark.py --generate --api-url http://localhost:8001
```

This will, for each non-`_ref` test folder:
1. POST the original image to the backendAPI (`/api/v1/personas`)
2. Fetch the rendered lego PNG (`/api/v1/personas/:id/image`)
3. Save it as `lego.png` in that folder (always overwrites)

Then run the benchmark as normal.

| Flag | Default | Description |
|---|---|---|
| `--generate` | off | Generate lego images before benchmarking |
| `--api-url` | `http://localhost:8001` | Base URL for the backendAPI |
| `--folder` | `./test_images` | Path to test images folder |

Reference folders (`_ref_*`) are never regenerated — they are expected to already have their `lego.*` files.

---

## Example Output

```
Loading CLIP model...
Running on: cpu
Loading DeepFace...

=== Reference Pairs ===
Name                     CLIP    Face   Color   Score  Relative  Rating
------------------------------------------------------------------------
_ref_alice             0.7823  0.6102  0.7541  0.7011  (baseline)  (combined)
_ref_bob               0.7541  0.5834  0.7203  0.6778  (baseline)  (combined)
------------------------------------------------------------------------
  Baseline (avg of 2 references): 0.6895

=== Test Pairs ===
Name                     CLIP    Face   Color   Score  Relative  Rating
------------------------------------------------------------------------
person1                0.7634  0.5901  0.7312  0.6821     98.9%  ✅ Good    (combined)
person2                0.7201  0.4900  0.6834  0.6184     89.7%  ⚠️  OK     (combined)
person3_wrong_lego     0.7455  0.2100  0.3201  0.4030     58.4%  ❌ Poor    (combined)
------------------------------------------------------------------------

Total test pairs:  3
Baseline score:    0.6895
Average relative:  82.3%
Weights: CLIP 25% / Face 50% / Color 25%

🟡 Pipeline result: ACCEPTABLE
```

Notice how `person3_wrong_lego` (wrong person rendered) scores 58.4% — well into Poor — because the color histogram catches the mismatch even when CLIP misses it.

---

## Score Interpretation

| Relative Score | Rating | Meaning |
|---|---|---|
| ≥ 90% of baseline | ✅ Good | On par with your golden examples |
| 70–90% of baseline | ⚠️ OK | Noticeably worse but acceptable |
| < 70% of baseline | ❌ Poor | Significantly below your standard |

---

## Tips

- **More references = more stable baseline.** 3–5 diverse references is ideal.
- **Pick diverse references** — different hair, skin tone, age — so the baseline reflects general capability.
- **Re-run after every pipeline change** and compare the average relative score to catch regressions.
- **Tweak weights** in the script if one signal matters more to you — the `CLIP_WEIGHT`, `FACE_WEIGHT`, `COLOR_WEIGHT` constants are at the top of the scoring section.

---

## Troubleshooting

**`git` not found** — Install Git from [git-scm.com](https://git-scm.com), required to install CLIP.

**Face score shows `n/a`** — DeepFace failed on those images. Script falls back to CLIP+color automatically.

**No reference folders found** — Name your golden example folders starting with `_ref` (e.g. `_ref_alice`).

**Out of memory** — Reduce test set size.