# Training Your Own Fashion Model — Learning Path

Your current code (`main.py`, `train.py`) jumps straight into advanced territory: CLIP, fine-tuning, 
custom heads, two-phase training. This guide walks you through every building block so you understand 
WHY each piece exists before you use it.

Run the files in order. Each one is self-contained and prints output so you can see exactly what's happening.

---

## The Big Picture

Here's what your final model does, top to bottom:

```
Photo of outfit
      ↓
[Image → Tensor]         Convert pixels to numbers a model can read
      ↓
[CLIP Vision Encoder]    Compress the image into a 512-number "fingerprint"
      ↓
[Your Classification Heads]  Turn that fingerprint into predictions
      ↓
category=jeans, color=blue, style=wide-leg
      ↓
Search query → Product links
```

The part YOU train is the classification heads. CLIP already knows how to "understand" images —
you're teaching it what categories matter for *your* use case.

---

## Step-by-Step Files

| File | What you learn | Run time |
|------|---------------|----------|
| `learn/01_tensors_and_images.py` | How images become numbers | < 5 sec |
| `learn/02_training_loop.py` | The core loop every model uses | < 10 sec |
| `learn/03_dataset.py` | Loading your own images + labels | < 5 sec |
| `learn/04_clip_zero_shot.py` | What main.py actually does | ~30 sec (downloads CLIP) |
| `learn/05_fine_tune.py` | Training classification heads | needs dataset |
| `learn/06_evaluate.py` | Measuring if your model is good | needs trained model |

---

## Key Concepts Cheat Sheet

**Tensor** — a multi-dimensional array of numbers. An image is a tensor of shape `[3, 224, 224]`
(3 color channels × 224 pixels high × 224 pixels wide).

**Forward pass** — feeding input through the model to get a prediction.

**Loss** — a single number measuring how wrong the prediction was. Lower = better.

**Backward pass (backprop)** — computing how much each weight contributed to the loss.

**Optimizer step** — nudging each weight slightly in the direction that reduces loss.

**Epoch** — one full pass through your entire training dataset.

**Overfitting** — the model memorizes training data but fails on new images.
Fix: more data, dropout, data augmentation, or early stopping.

**Transfer learning** — starting from a model already trained on millions of images (CLIP)
instead of random weights. This is why you can get good results with ~5k images instead of millions.

**Fine-tuning** — taking a pre-trained model and training it a bit more on your specific data.
Two phases:
  1. Freeze the backbone, train only your new heads (fast, stable)
  2. Unfreeze the backbone, train everything with a tiny learning rate (squeezes out extra accuracy)

---

## Why CLIP Specifically?

CLIP (Contrastive Language-Image Pre-training) was trained on 400 million image-text pairs scraped 
from the internet. It learned to match images with descriptions like "a blue oversized hoodie."

This means it already has strong visual features for clothing — colors, textures, shapes, styles.
You're not teaching it what "blue" looks like from scratch. You're just adding a small layer that 
says "given these features, predict our specific category labels."

Zero-shot (what main.py does now):
  - No training needed
  - You describe categories in plain English
  - CLIP scores how well the image matches each description
  - Accurate enough to start, but limited by how well your text descriptions match the images

Fine-tuned (what train.py builds toward):
  - You show it thousands of labeled fashion images
  - It learns the exact visual patterns that matter for YOUR categories
  - More accurate, especially for edge cases

---

## Data You Need

To train, you need labeled images. Best free option: **Myntra Fashion Dataset** on Kaggle.
~44k images with category, color, gender labels. Your `train.py` is already written for it.

Download: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
Unzip to: `../data/myntradataset/` (next to the backend folder)

Expected structure:
```
data/myntradataset/
  images/          ← 44k .jpg files named by product ID
  styles.csv       ← labels: id, gender, articleType, baseColour, ...
```

---

## Running Order

```bash
cd backend

# Install deps if needed
pip install torch torchvision transformers pillow

# Work through the lessons
python learn/01_tensors_and_images.py
python learn/02_training_loop.py
python learn/03_dataset.py
python learn/04_clip_zero_shot.py   # needs internet on first run

# Once you have the Myntra dataset downloaded:
python learn/05_fine_tune.py        # ~1-2 hrs on CPU, ~10 min on GPU
python learn/06_evaluate.py         # run after fine-tuning
```

After `05_fine_tune.py` completes, `data/shein_model.pt` exists and `main.py` will automatically
use your trained model instead of pure zero-shot.
