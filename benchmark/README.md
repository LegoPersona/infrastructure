# Lego Persona Benchmark

An automated tool that measures how well the Lego persona pipeline represents the original person, scored relative to your own golden reference examples.

---

## How It Works

The benchmark uses **three signals** combined into a single score:

| Signal | What it measures | Weight |
|---|---|---|
| Attribute match | Did the pipeline pick the right hair color, eye color, facial hair, etc.? | 60% |
| Visual similarity | Overall appearance match between original and Lego image | 20% |
| Color match | Do the dominant hue and saturation distributions align? | 20% |

### Attribute match (60%)

The core metric. For each of 7 appearance features, CLIP zero-shot classifies both the original photo and the Lego figure, then a dot-product between the two probability distributions measures agreement.

The key design: **separate prompt sets per domain**. The original is classified with person-specific prompts (`"a photo of a person with black hair"`), the Lego with Lego-specific prompts (`"a lego minifigure with black hair"`). This avoids forcing cross-domain text matching and gives CLIP the right visual context for each image type.

Attribute categories and their weights within the 60% are derived directly from the lego-service template options:

| Attribute | Weight |
|---|---|
| `hair_color` | 25% |
| `hair_style` | 20% |
| `facial_hair` | 20% |
| `eye_color` | 15% |
| `eyebrow_color` | 10% |
| `eyebrow_shape` | 5% |
| `nose_shape` | 5% |

### Visual similarity (20%)

CLIP ViT-L/14 cosine similarity between the two image embeddings. Raw CLIP scores have a floor of ~0.30 for any two real-world images (random pairs don't score near zero), so scores are rescaled: `max(0, (raw - 0.30) / 0.70)`. This gives meaningful spread rather than everything clustering around 0.5.

### Color match (20%)

HSV color histograms compared using the Bhattacharyya coefficient. H (hue) and S (saturation) channels are scored independently and averaged — this prevents a saturation match alone from inflating the score when the hues are completely different (e.g. a solid-colored image matching a Lego on saturation alone).

### Reference-based scoring

Rather than fixed absolute thresholds, the benchmark calibrates against your own golden examples — pairs stored in `_ref` folders that you've decided look good.

1. Score all `_ref` folders → average them → **baseline**
2. Score all test folders → express each as a **% of baseline**
3. ≥ 90% = Good, 70–90% = OK, < 70% = Poor

---

## Technology

- **[CLIP ViT-L/14](https://github.com/openai/CLIP)** — OpenAI's contrastive image-language model. Used for both visual similarity (image embeddings) and attribute classification (zero-shot text-image matching). ViT-L/14 is the large variant with 14px patch grid, giving better fine-grained discrimination than the smaller ViT-B/32.
- **PyTorch** — model inference backend
- **NumPy / Pillow** — image processing and histogram computation

No face recognition or deep-face models are used. All scoring runs through CLIP alone, which handles the cross-domain gap between real photos and stylized Lego figures.

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
        _ref_bob/
            original.jpg
            lego.jpg
        person1/            ← test case
            original.jpg
            lego.jpg        ← can be generated with --generate
        person2/
            original.jpg
            lego.jpg
```

- Any folder starting with `_ref` is a reference (used to build the baseline)
- Everything else is a test case scored against the baseline
- Both `.jpg` and `.png` are supported

---

## Setup

```bash
pip install -r requirements.txt
```

First run downloads CLIP ViT-L/14 weights (~890MB), cached after that.

---

## Usage

```bash
# Basic run
python benchmark.py

# Custom folder
python benchmark.py --folder ./my_test_set

# Show per-attribute score breakdown for each pair
python benchmark.py --verbose

# Generate lego images from originals via the live pipeline, then benchmark
python benchmark.py --generate --api-url http://localhost:3000

# Generate with API authentication
python benchmark.py --generate --api-url http://localhost:3000 --token <your_access_token>
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--folder` | `./test_images` | Path to test images folder |
| `--generate` | off | Call the backendAPI to generate lego images before scoring |
| `--api-url` | `http://localhost:3000` | Base URL for the backendAPI |
| `--token` | none | Bearer access token sent as `Authorization: Bearer <token>` on generation requests |
| `--verbose` | off | Print per-attribute scores under each pair |

`--generate` POSTs each `original.*` to `/api/v1/personas`, fetches the rendered image from `/api/v1/personas/:id/image`, and saves it as `lego.png`. Reference folders are never regenerated.

After all generations complete, a token usage summary is printed:

```
[generate] Token usage across 3 generation(s):
  Total  — input: 4200  output: 810  total: 5010
  Avg    — input: 1400.0  output: 270.0  total: 1670.0
```

---

## Score Interpretation

| Relative score | Rating | Meaning |
|---|---|---|
| ≥ 90% of baseline | ✅ Good | On par with your golden examples |
| 70–90% of baseline | ⚠️ OK | Noticeably worse but acceptable |
| < 70% of baseline | ❌ Poor | Significantly below your standard |

---

## Tips

- **More references = more stable baseline.** 3–5 diverse references is ideal.
- **Pick diverse references** — different hair color, style, and facial features — so the baseline reflects general capability rather than one specific look.
- **Re-run after every pipeline change** and compare the average relative score to catch regressions.
- **Use `--verbose`** when debugging a specific pair to see which attributes are mismatching.
- **No reference folders found** — name your golden example folders starting with `_ref` (e.g. `_ref_alice`).
