"""
LESSON 5: Fine-Tuning CLIP — Training Your Own Fashion Classifier

This file is BOTH a teaching document AND a working training script.
Read the printed chapters at startup, then watch training happen live.

Prerequisites:
  1. Myntra dataset extracted to:  ../../data/myntradataset/
     Get it: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
     After download, you should have:
       data/myntradataset/styles.csv
       data/myntradataset/images/12345.jpg  (40,000+ images)

  2. Dependencies:
     pip install torch torchvision transformers pillow

Run:
  python learn/05_fine_tune.py
  python learn/05_fine_tune.py --epochs 20 --batch-size 64

Training time:
  CPU (MacBook):  ~2-4 hours for 15 epochs
  GPU (Colab):    ~15-25 minutes — use Google Colab with T4 GPU (free)
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ══════════════════════════════════════════════════════════════════════════════
# EDUCATIONAL CHAPTERS  (printed at startup before training begins)
# ══════════════════════════════════════════════════════════════════════════════

def print_chapter_1():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1: What Is Transfer Learning?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Imagine you already know how to drive a car.
Now someone asks you to drive a delivery truck.

You don't start from zero. You already know:
  ✓ How to steer
  ✓ What traffic lights mean
  ✓ How brakes and acceleration work

You just need to learn the truck-specific stuff:
  → The truck is wider (different lane judgment)
  → The gear shifts differently
  → You need to judge height for bridges

Transfer learning works exactly this way in ML.

CLIP already knows (from 400 million images):
  ✓ What colors look like
  ✓ What textures look like (denim, cotton, leather, knit...)
  ✓ What shapes correspond to what clothing words
  ✓ How to encode an image into a 512-d meaning vector

We're just teaching it the "truck-specific stuff":
  → What YOUR specific category names mean (jeans, blouse, skirt...)
  → YOUR specific color groupings
  → YOUR dataset's visual style

PARAMETER COUNTS (let's look at the actual numbers):
""")

    # Show parameter counts
    try:
        import torch
        import torch.nn as nn
        from transformers import CLIPModel

        print("  Loading CLIP to count parameters...")
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        vision_params = sum(p.numel() for p in clip.vision_model.parameters())
        proj_params   = sum(p.numel() for p in clip.visual_projection.parameters())

        # Sample head (category: 10 classes, color: 13 classes)
        head_cat = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,10))
        head_col = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,13))
        head_params = sum(p.numel() for p in head_cat.parameters()) + \
                      sum(p.numel() for p in head_col.parameters())
        backbone_total = vision_params + proj_params

        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Component                       Parameters            │
  ├─────────────────────────────────────────────────────────┤
  │  CLIP vision encoder (ViT-B/32)  {vision_params:>14,}  │
  │  CLIP visual projection          {proj_params:>14,}  │
  │  Our category head               {sum(p.numel() for p in head_cat.parameters()):>14,}  │
  │  Our color head                  {sum(p.numel() for p in head_col.parameters()):>14,}  │
  ├─────────────────────────────────────────────────────────┤
  │  Backbone total                  {backbone_total:>14,}  │
  │  Our heads total                 {head_params:>14,}  │
  │  Our heads as % of total         {head_params/(backbone_total+head_params)*100:>13.2f}% │
  └─────────────────────────────────────────────────────────┘

  We're adding {head_params/(backbone_total+head_params)*100:.1f}% of new parameters on top of a
  fully-trained model. This is why training is fast and works well
  even with only tens of thousands of examples.

  An analogy in numbers:
    Training from scratch = lifting 86 million kg
    Fine-tuning           = lifting 50,000 kg  (after someone
                            already did the 86 million kg work)
""")
        del clip
    except Exception as e:
        print(f"""
  CLIP backbone:                ~86,192,000 parameters
  CLIP visual projection:          262,656 parameters
  Our classification heads:        ~50,000 parameters

  Our heads are only ~0.06% of the total.
  We borrow the backbone's knowledge and add a tiny translator.

  (Could not load CLIP to show live counts: {e})
""")


def print_chapter_2():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2: What Are Classification Heads?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A "head" is a small neural network bolted onto the OUTPUT of a large
pretrained model. It takes the pretrained model's rich features and
maps them to YOUR specific categories.

Think of it as a custom translator:
  CLIP speaks "generic visual language" → Head translates to "fashion terms"

OUR MODEL ARCHITECTURE:

  Input image (224×224 RGB)
         │
         ▼
  ┌─────────────────────────────────────┐
  │  CLIP Vision Encoder                │
  │  (ViT: splits image into 14×14      │
  │   patches, applies Transformer)     │
  └──────────────┬──────────────────────┘
                 │  [1, 768] raw visual features
                 ▼
  ┌─────────────────────────────────────┐
  │  visual_projection                  │
  │  Linear(768 → 512)                  │
  └──────────────┬──────────────────────┘
                 │  [1, 512] embedding
                 ▼
           L2 Normalize
           (makes length = 1.0)
                 │
       ┌─────────┴──────────┐
       │                    │
       ▼                    ▼
  ┌──────────────┐    ┌──────────────┐
  │ Category     │    │ Color        │  ← You can add more heads!
  │ Head         │    │ Head         │     (style, sleeve, neckline...)
  │              │    │              │
  │ Linear(512→256)   Linear(512→256)│
  │ ReLU         │    │ ReLU         │
  │ Dropout(0.2) │    │ Dropout(0.2) │
  │ Linear(256→N)│    │ Linear(256→M)│
  └──────┬───────┘    └──────┬───────┘
         │                   │
         ▼                   ▼
   N logit scores       M logit scores
   (one per category)   (one per color)

  N = number of categories (e.g. 10: jeans, dress, top...)
  M = number of colors     (e.g. 13: blue, red, black...)

WHAT IS A LOGIT?
  A "logit" is just a raw unnormalized score.
  Higher = more confident it's that class.
  To get probabilities: apply softmax(logits).

EACH COMPONENT EXPLAINED:

  Linear(512, 256):
    A matrix multiplication. 512 inputs × 256 outputs.
    Learns which combinations of CLIP features matter for fashion.

  ReLU (Rectified Linear Unit):
    f(x) = max(0, x)
    Replaces negative values with 0. Without this, stacking linear layers
    is mathematically equivalent to ONE linear layer — useless.
    ReLU adds non-linearity so the network can learn complex patterns.

    Example:
""")
    # Show ReLU with actual numbers
    import math
    test_inputs = [-2.3, -0.8, 0.0, 0.5, 1.7, 3.2]
    print("    Input  →  After ReLU")
    for x in test_inputs:
        relu_x = max(0, x)
        print(f"    {x:>5.1f}  →  {relu_x:.1f}")
    print("""
  Dropout(0.2):
    Randomly sets 20% of neuron outputs to 0 during training.
    WHY: Prevents the model from relying too heavily on any single neuron.
    Covered in depth in Chapter 3.

  Linear(256, N):
    Final mapping from 256 intermediate features to N class scores.
    These N numbers are the logits — one per category.
""")


def print_chapter_3():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3: Dropout — Why Randomness Prevents Overfitting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: OVERFITTING
────────────────────
  Imagine you study for an exam by memorizing exact questions from
  last year's paper. You score 100% on those questions.
  But on this year's DIFFERENT questions, you fail.

  This is overfitting: the model memorizes the training data instead of
  learning generalizable patterns.

  You know it's happening when:
    Training accuracy: 98%
    Validation accuracy: 62%    ← big gap = overfitting

DROPOUT AS A SOLUTION
──────────────────────
  During TRAINING: randomly zero out 20% of neurons on every forward pass.
  During INFERENCE (model.eval()): all neurons are active.

  WHY does this help?
  Each training step, a different 20% is disabled. The model CANNOT
  rely on any single neuron always being there. It must spread knowledge
  across many neurons → more robust, generalizable features.

  ANALOGY: If you study with a study group but each session one person
           is randomly absent, you learn to understand the material
           yourself instead of just copying answers from that person.

Let's show the stochastic behavior concretely:
""")

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(42)
        layer = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.2))

        dummy_input = torch.ones(1, 8) * 2.0

        print("  Same input → different outputs during training (Dropout is ON):\n")
        layer.train()  # Dropout is active
        for run in range(4):
            out = layer(dummy_input)
            vals = [f"{v:.3f}" for v in out[0].tolist()]
            print(f"  Run {run+1}: [{', '.join(vals)}]")

        print()
        layer.eval()  # Dropout is OFF
        out = layer(dummy_input)
        vals = [f"{v:.3f}" for v in out[0].tolist()]
        print(f"  Same input during inference (Dropout OFF, deterministic):")
        print(f"  Result: [{', '.join(vals)}]")
        print()
        print("  See how training outputs differ each run (zeros randomly appear)?")
        print("  Inference output is always the same (no zeros = full capacity).")
    except Exception as e:
        print(f"  (Could not run live demo: {e})")
        print("  Conceptually: training passes show random zeros; inference is deterministic.")

    print("""
  OUR CHOICE: 0.2 (zero out 20%)
  ─────────────────────────────
  Too low (0.05): barely any regularization, still overfits
  Just right (0.2): enough randomness to prevent memorization
  Too high (0.5): too much information lost, model can't learn at all

  0.2–0.5 is the typical range. We use 0.2 because our heads are small.
  Higher dropout is more useful in larger networks.
""")


def print_chapter_4():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4: Two-Phase Training — Why You Can't Just Train Everything at Once
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1: Freeze backbone, train heads only
──────────────────────────────────────────
  "Frozen" means: we do NOT update these weights during training.
  requires_grad = False → PyTorch won't compute gradients for these layers.

  WHY start frozen?

  Think of it this way: the heads start with RANDOM weights.
  They output garbage at the start.

  If you also update the backbone with these garbage gradients, you'd
  corrupt the 86M parameters CLIP spent months learning. It's like
  hiring someone who just started their first day and immediately
  giving them override access to the entire codebase.

  Instead: freeze the backbone. Train only the heads for the first half.
  Let the heads stabilize. Then the gradients they produce are meaningful.

  Phase 1 shows the gradient status like this:

  backbone layer 1:   requires_grad = False  ← frozen
  backbone layer 2:   requires_grad = False  ← frozen
  ...
  head.linear1.weight: requires_grad = True  ← training
  head.linear2.weight: requires_grad = True  ← training

PHASE 2: Unfreeze backbone, fine-tune everything
────────────────────────────────────────────────
  Now the heads are stable (not outputting garbage).
  We unfreeze the backbone and train everything — but with VERY different
  learning rates:

    Heads:    learning rate = 1e-3  (0.001)   ← normal pace
    Backbone: learning rate = 1e-5  (0.00001) ← 100× slower

  WHY such a small lr for the backbone?

  The backbone already works great. We want gentle nudges to make it
  more fashion-aware, not massive updates that destroy what it learned.

  ANALOGY: You're an expert chef (the backbone). A new restaurant
  hires you (fine-tuning). They want slight adjustments to your style
  (smaller plates, different spice levels). They adjust you SLOWLY
  so you don't forget all your training. They don't fire you and
  rebuild you from scratch.

  The 100× ratio (1e-3 vs 1e-5) is a well-tested heuristic.
  The backbone needs orders-of-magnitude smaller updates than new layers.

  Phase 2 shows:

  >>> Phase 2: Unfreezing CLIP backbone (lower lr) <<<

  backbone layer 1:   requires_grad = True  ← now training (lr=1e-5)
  backbone layer 2:   requires_grad = True  ← now training (lr=1e-5)
  ...
  head.linear1.weight: requires_grad = True  ← still training (lr=1e-3)
""")


def print_chapter_5(csv_path, category_map, color_map):
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5: The Data Loading Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Myntra dataset (styles.csv) has rows like:

  id,gender,masterCategory,subCategory,articleType,baseColour,...
  15970,Men,Apparel,Topwear,Shirts,Navy Blue,...
  39386,Women,Apparel,Bottomwear,Jeans,Blue,...
  59263,Women,Apparel,Topwear,Tops,White,...

Each id maps to an image file:  images/15970.jpg

WE FILTER to Women/Girls clothing only.
WHY? The app is for fashion finding. Menswear has fewer overlap categories.
     This is a DESIGN DECISION you can change.

CATEGORY_MAP — grouping Myntra's specific labels into our app's categories:
""")

    print("  Myntra articleType     → Our category")
    print("  " + "─"*40)
    for myntra_name, our_name in list(category_map.items())[:12]:
        print(f"  {myntra_name:<22} → {our_name}")
    print(f"  ... ({len(category_map)} total mappings)")

    print("""
  WHY GROUP? Myntra has 100+ articleTypes. Our app only needs ~10 categories.
  Grouping reduces noise and gives more training examples per class.

  DESIGN CHOICE: These mappings are YOURS to control.
  If you want "Shirts" to be its own category, change the map.
  The model learns whatever taxonomy you give it.

COLOR_MAP — normalizing Myntra's specific color names to our 13 colors:
""")
    print("  Myntra baseColour      → Our color")
    print("  " + "─"*40)
    for myntra_name, our_name in list(color_map.items())[:10]:
        print(f"  {myntra_name:<22} → {our_name}")
    print(f"  ... ({len(color_map)} total mappings)")

    print("""
  WHY GROUP COLORS? "Navy Blue", "Teal", "Cerulean" are all blue to a shopper.
  Grouping gives more training examples per color class.

WHAT HAPPENS WITH UNKNOWN ROWS:
  If a row has an articleType not in CATEGORY_MAP → skip it.
  If a row has a baseColour not in COLOR_MAP → skip it.
  If the image file doesn't exist → skip it.
  This is how real ML works: messy data, we filter aggressively.
""")

    if csv_path.exists():
        print("  Scanning styles.csv to show category distribution...\n")
        cat_counts = {}
        col_counts = {}
        total_skipped = 0
        total_kept    = 0
        with open(csv_path, errors="ignore") as f:
            for row in csv.DictReader(f):
                if row.get("gender") not in ("Women", "Girls"):
                    total_skipped += 1
                    continue
                cat = category_map.get(row.get("articleType",""), "")
                col = color_map.get(row.get("baseColour",""), "")
                if not cat or not col:
                    total_skipped += 1
                    continue
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                col_counts[col] = col_counts.get(col, 0) + 1
                total_kept += 1

        print(f"  {'Category':<15} | {'Count':>6} | Bar")
        print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*25}")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (count // 50)
            print(f"  {cat:<15} | {count:>6} | {bar}")

        print()
        print(f"  {'Color':<10} | {'Count':>6} | Bar")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*25}")
        for col, count in sorted(col_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (count // 50)
            print(f"  {col:<10} | {count:>6} | {bar}")
        print()
        print(f"  Total usable: {total_kept:,}    Skipped: {total_skipped:,}")
    else:
        print("  (Dataset not found — will show live counts when training runs)")


def print_chapter_6():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 6: Gradient Clipping — Preventing Explosions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To understand gradient clipping, we need to understand gradients first.

WHAT IS A GRADIENT?
  A gradient tells the optimizer: "move this weight in this direction
  by this much to reduce the loss."

  If the gradient for a weight is +2.3, the optimizer reduces the weight.
  If it's -0.4, the optimizer increases it.
  The MAGNITUDE tells how strongly to move.

THE EXPLOSION PROBLEM:
  Sometimes gradients become astronomically large.
  Example: gradient = 10,000 → the optimizer makes a huge jump.
  The model overshoots the optimal weights entirely.
  Next step: gradient = -15,000 in the other direction. Chaos.

  This is called "gradient explosion." It causes loss to spike wildly
  and training to diverge (go haywire).

  It happens most often during fine-tuning large backbones because
  the backbone has very deep layers — gradients multiply through many
  layers and can compound into large values.

GRADIENT CLIPPING SOLUTION:
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

  This computes the TOTAL gradient magnitude across ALL parameters
  (called the "global norm"). If it exceeds max_norm=1.0, it scales
  ALL gradients DOWN proportionally so the total is exactly 1.0.

  ANALOGY: You're biking downhill and going too fast.
           You don't just brake one wheel — you apply brakes evenly
           so the bike stays stable.

  Show: gradient norms before and after clipping
""")

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        net = nn.Linear(4, 4)

        # Manually set extreme gradients to demonstrate
        for param in net.parameters():
            param.grad = torch.randn_like(param) * 50  # simulate large gradients

        total_norm_before = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5

        print(f"  Gradient norm BEFORE clipping: {total_norm_before:.2f}")

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        total_norm_after = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        print(f"  Gradient norm AFTER  clipping: {total_norm_after:.4f}  (capped at 1.0)")
        print()
        scale = total_norm_after / total_norm_before
        print(f"  All gradients scaled by: {scale:.4f}")
        print(f"  Individual gradients are smaller, but their RATIO is preserved.")
        print(f"  The optimizer still moves in the RIGHT direction, just more gently.")
    except Exception as e:
        print(f"  Example with simulated large gradients:")
        print(f"    Gradient norm before clipping: 87.34")
        print(f"    Gradient norm after  clipping:  1.00")
        print(f"    Scale factor applied: 0.01145  (all gradients ÷ 87.34)")
        print(f"  (Live demo unavailable: {e})")

    print()


def print_chapter_7():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 7: Reading the Training Output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When training begins, you'll see a table like this:

  Epoch | Phase | Train Loss | Val Loss | Category Acc | Color Acc
  ──────────────────────────────────────────────────────────────────
      1 |   1   |   2.1503   |  2.0412  |   28%        | 19%
      2 |   1   |   1.8321   |  1.7891  |   41%        | 33%
      3 |   1   |   1.6105   |  1.5982  |   54%        | 47%
     ...
      8 |   2   |   0.9401   |  0.9812  |   72%        | 68%
     ...
     15 |   2   |   0.5821   |  0.6102  |   84%        | 79%

WHAT EACH COLUMN MEANS:

  Epoch      : Training iteration. One epoch = all training images seen once.

  Phase      : 1 = backbone frozen, heads only.
               2 = backbone unfrozen, everything training.
               You'll see numbers drop faster in Phase 2.

  Train Loss : How wrong the model is on TRAINING images.
               CrossEntropyLoss. Lower = better. Starts ~2.3 for 10 classes.
               (why 2.3? log(10) ≈ 2.3 — that's random-guess loss)

  Val Loss   : How wrong the model is on VALIDATION images it never trained on.
               This is the number that matters for real-world performance.

  Category Acc: What % of validation images got the right category.
                28% at epoch 1 is fine — random chance for 10 classes = 10%.

  Color Acc  : What % of validation images got the right color.
               13 classes → random baseline is ~8%.

GOOD SIGNS:
  ✓ Both train loss AND val loss dropping together
  ✓ Val loss sometimes equals or beats train loss (good generalization)
  ✓ Accuracy climbing steadily through Phase 1 and Phase 2

WARNING SIGNS:
  ⚠ Val loss goes UP while train loss goes DOWN → overfitting
      Fix: add more dropout, add more data augmentation, reduce epochs

  ⚠ Both losses barely move → learning rate too small, or data issue
      Fix: increase lr, check dataset loaded correctly

  ⚠ Loss jumps wildly between epochs → learning rate too large
      Fix: reduce lr, or make sure gradient clipping is on

  ⚠ Big gap at Phase 2 transition → backbone being updated too fast
      Fix: reduce --lr-backbone (default 1e-5 is already conservative)

WHEN TO STOP:
  Stop when val loss hasn't improved for 3-5 epochs in a row.
  The model is saved automatically each time val loss improves.
  So you can always kill training early — the best checkpoint is saved.

EXPECTED FINAL ACCURACY (with Myntra ~20K fashion images):
  Category: ~80-90%  (depends on class balance)
  Color:    ~75-85%
  These are much better than zero-shot (~60-70% on fashion-specific categories).
""")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "myntradataset"
IMAGES_DIR  = DATASET_DIR / "images"
CSV_PATH    = DATASET_DIR / "styles.csv"

OUTPUT_DIR  = Path(__file__).parent.parent / "data"
MODEL_PATH  = OUTPUT_DIR / "shein_model.pt"
LABELS_PATH = OUTPUT_DIR / "label_maps.json"

# ─── Label mappings ───────────────────────────────────────────────────────────
# These are DESIGN DECISIONS. You control what categories the model learns.
# Myntra has 100+ articleTypes. We group them into ~10 categories we care about.

CATEGORY_MAP = {
    "Tops":        "top",
    "Tshirts":     "top",
    "Camisoles":   "top",
    "Shirts":      "blouse",
    "Kurtis":      "blouse",
    "Tunics":      "blouse",
    "Dresses":     "dress",
    "Nightdress":  "dress",
    "Jeans":       "jeans",
    "Skirts":      "skirt",
    "Shorts":      "shorts",
    "Capris":      "shorts",
    "Trousers":    "trousers",
    "Leggings":    "trousers",
    "Track Pants": "trousers",
    "Sweaters":    "sweater",
    "Sweatshirts": "sweater",
    "Jackets":     "jacket",
}

COLOR_MAP = {
    "Black":     "black",
    "White":     "white",
    "Off White": "white",
    "Cream":     "white",
    "Blue":      "blue",
    "Navy Blue": "blue",
    "Pink":      "pink",
    "Peach":     "pink",
    "Magenta":   "pink",
    "Lavender":  "purple",
    "Purple":    "purple",
    "Green":     "green",
    "Red":       "red",
    "Maroon":    "red",
    "Burgundy":  "red",
    "Grey":      "gray",
    "Brown":     "brown",
    "Beige":     "beige",
    "Khaki":     "beige",
    "Yellow":    "yellow",
    "Orange":    "orange",
    "Multi":     "multicolor",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Read styles.csv and return a list of dicts, one per usable image.
    Filters to Women/Girls, keeps only rows with known category AND color,
    and verifies each image file actually exists on disk.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"\n  Dataset not found at: {DATASET_DIR}\n"
            "  Download: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small\n"
            "  Extract so you have:  data/myntradataset/styles.csv\n"
            "                        data/myntradataset/images/12345.jpg"
        )

    rows = []
    skipped_gender   = 0
    skipped_category = 0
    skipped_color    = 0
    skipped_missing  = 0

    print("  Reading styles.csv...")
    with open(CSV_PATH, errors="ignore") as f:
        for row in csv.DictReader(f):
            if row.get("gender") not in ("Women", "Girls"):
                skipped_gender += 1
                continue

            category = CATEGORY_MAP.get(row.get("articleType", ""))
            color    = COLOR_MAP.get(row.get("baseColour", ""))

            if not category:
                skipped_category += 1
                continue
            if not color:
                skipped_color += 1
                continue

            img_path = IMAGES_DIR / f"{row['id']}.jpg"
            if not img_path.exists():
                skipped_missing += 1
                continue

            rows.append({
                "image_path": str(img_path),
                "category":   category,
                "color":      color,
            })

    print(f"  Usable samples: {len(rows):,}")
    print(f"  Skipped — wrong gender:    {skipped_gender:,}")
    print(f"  Skipped — unknown category:{skipped_category:,}")
    print(f"  Skipped — unknown color:   {skipped_color:,}")
    print(f"  Skipped — image missing:   {skipped_missing:,}")
    return rows


def build_label_maps(rows):
    """
    Create integer index mappings for each prediction field.
    The model outputs integers (0, 1, 2...) — this maps them back to names.

    Example output:
      {
        "category": {"blouse": 0, "dress": 1, "jeans": 2, "skirt": 3, ...},
        "color":    {"beige": 0, "black": 1, "blue": 2, ...}
      }
    """
    categories = sorted(set(r["category"] for r in rows))
    colors     = sorted(set(r["color"]    for r in rows))
    maps = {
        "category": {name: i for i, name in enumerate(categories)},
        "color":    {name: i for i, name in enumerate(colors)},
    }
    print(f"\n  Categories ({len(categories)}): {categories}")
    print(f"  Colors     ({len(colors)}): {colors}")
    return maps


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FashionDataset(Dataset):
    """
    PyTorch Dataset — teaches the DataLoader how to load one sample.

    The DataLoader calls __getitem__(idx) repeatedly to build batches.
    It handles shuffling, batching, and parallel loading automatically.

    __len__: how many total samples are there?
    __getitem__: give me sample number idx (image tensor + labels)
    """

    def __init__(self, rows, label_maps, transform):
        self.rows       = rows
        self.label_maps = label_maps
        self.transform  = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # Load image from disk
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            # Corrupted image — return a black placeholder rather than crash
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Apply transforms (resize, crop, normalize, augment)
        image = self.transform(image)

        # Convert string labels to integers for the model
        # Model outputs integer indices → we decode back to names at inference
        labels = {
            "category": self.label_maps["category"][row["category"]],
            "color":    self.label_maps["color"][row["color"]],
        }
        return image, labels


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class FashionClassifier(nn.Module):
    """
    CLIP backbone + classification heads.

    Combines Chapters 1-3 into one working module.
    Backbone: 86M parameters (CLIP's learned visual knowledge).
    Heads:    ~50K parameters (our fashion taxonomy translator).
    """

    def __init__(self, label_maps, clip_model):
        super().__init__()
        # Borrow CLIP's vision components
        self.clip_vision = clip_model.vision_model
        self.clip_proj   = clip_model.visual_projection

        embed_dim = 512  # CLIP ViT-B/32 projects to 512-d

        # One head per prediction field
        # nn.ModuleDict makes PyTorch aware of all submodules for .parameters()
        # Without this, the heads' parameters wouldn't be included in optimization
        self.heads = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(embed_dim, 256),  # compress 512 → 256
                nn.ReLU(),                  # non-linearity
                nn.Dropout(0.2),            # regularization
                nn.Linear(256, len(lmap)),  # 256 → N class scores
            )
            for field, lmap in label_maps.items()
        })

    def forward(self, pixel_values):
        """
        One forward pass:
          1. Extract visual features via ViT (768-d)
          2. Project to shared embedding space (512-d)
          3. L2 normalize (unit sphere)
          4. Run through each head → logits per field
        """
        vision_out = self.clip_vision(pixel_values=pixel_values)
        features   = self.clip_proj(vision_out.pooler_output)  # [B, 512]

        # Normalize so embeddings live on unit sphere
        # (matches how CLIP was trained — keeps backbone features meaningful)
        features = features / features.norm(dim=-1, keepdim=True)

        # Each head gets the same 512-d features
        # Returns dict: {"category": [B, N_cat], "color": [B, N_col]}
        return {field: head(features) for field, head in self.heads.items()}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(epochs, batch_size, lr_head, lr_backbone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cpu":
        print("  Tip: Training on CPU is slow (~2-4 hours).")
        print("  Use Google Colab (free T4 GPU) for ~15-25 minutes instead.")
        print("  Instructions: https://colab.research.google.com/")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading dataset...")
    rows = load_data()
    if len(rows) < 100:
        print("  Not enough usable data. Check the dataset path.")
        return

    label_maps = build_label_maps(rows)

    # Save label maps — main.py needs these to decode predictions at inference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"\n  Label maps saved to: {LABELS_PATH}")

    # ── Transforms ────────────────────────────────────────────────────────────
    # CLIP expects 224×224 images normalized with specific mean/std
    # These values come from how CLIP was originally trained — we must match them
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_size = processor.image_processor.size["shortest_edge"]  # 224

    clip_mean = processor.image_processor.image_mean  # [0.481, 0.457, 0.408]
    clip_std  = processor.image_processor.image_std   # [0.268, 0.261, 0.275]

    # Training transforms include augmentation (random flips, color jitter)
    # WHY? Each epoch, the same image looks slightly different.
    # The model sees more variety → generalizes better (less overfitting).
    train_transform = transforms.Compose([
        transforms.Resize(clip_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.CenterCrop(clip_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    # Validation transforms: NO augmentation
    # WHY? We want consistent, reproducible measurements on the same images.
    val_transform = transforms.Compose([
        transforms.Resize(clip_size),
        transforms.CenterCrop(clip_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    # ── Dataset split ─────────────────────────────────────────────────────────
    # 90% training, 10% validation
    # Validation images are NEVER used during training — only to measure accuracy
    full_ds  = FashionDataset(rows, label_maps, train_transform)
    val_size = max(200, int(len(full_ds) * 0.10))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Apply val transform to the validation split
    # (DataLoader will call val_ds's transform for each batch)
    val_ds.dataset.transform = val_transform

    # num_workers=0: load images on main process (safer on macOS, avoids hangs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n  Training samples:   {train_size:,}")
    print(f"  Validation samples: {val_size:,}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Batches per epoch:  {len(train_loader)}")

    # ── Model setup ───────────────────────────────────────────────────────────
    print("\n  Loading CLIP backbone...")
    clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model     = FashionClassifier(label_maps, clip_base).to(device)

    # ── Phase 1: Freeze backbone ──────────────────────────────────────────────
    # requires_grad = False tells PyTorch: don't compute gradients for these layers
    # This means the optimizer will NOT update them (backbone stays frozen)
    print("\n  Phase 1: Freezing CLIP backbone (training heads only)\n")
    for p in model.clip_vision.parameters():
        p.requires_grad = False
    for p in model.clip_proj.parameters():
        p.requires_grad = False

    # Show which layers are trainable in Phase 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}  (heads only)")
    print(f"  Frozen parameters:    {frozen:,}  (CLIP backbone)")

    # Only pass head parameters to optimizer in Phase 1
    # CrossEntropyLoss: combines log-softmax + negative log likelihood
    # Standard choice for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.heads.parameters(),
        lr=lr_head,
        weight_decay=1e-4,   # L2 regularization: small penalty on large weights
    )

    # ReduceLROnPlateau: if val loss stops improving for 2 epochs, cut lr by 50%
    # WHY: when loss plateaus, the learning rate might be too large for fine adjustments
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, verbose=False
    )

    best_val_loss = float("inf")
    phase = 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  {'Epoch':>5} | Phase | {'Train Loss':>10} | {'Val Loss':>8} | {'Cat Acc':>7} | {'Col Acc':>7} | Time")
    print(f"  {'─'*5}-+-------+{'─'*11}+-{'─'*9}+-{'─'*8}+-{'─'*8}+{'─'*8}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Phase 2 transition ─────────────────────────────────────────────
        if epoch == epochs // 2 and phase == 1:
            phase = 2
            print(f"\n  >>> Phase 2 starts (epoch {epoch+1}): Unfreezing CLIP backbone with lr={lr_backbone} <<<\n")

            for p in model.clip_vision.parameters():
                p.requires_grad = True
            for p in model.clip_proj.parameters():
                p.requires_grad = True

            # Add backbone params with MUCH smaller lr
            # WHY separate param_group? AdamW allows per-group learning rates
            optimizer.add_param_group({
                "params": (list(model.clip_vision.parameters()) +
                           list(model.clip_proj.parameters())),
                "lr": lr_backbone,
                "weight_decay": 1e-4,
            })

            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters now: {trainable_now:,}  (backbone + heads)")
            print(f"  Head lr:     {lr_head:.0e}")
            print(f"  Backbone lr: {lr_backbone:.0e}  ({int(lr_head/lr_backbone)}× smaller)\n")
            print(f"  {'─'*5}-+-------+{'─'*11}+-{'─'*9}+-{'─'*8}+-{'─'*8}+{'─'*8}")

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()   # activates Dropout, enables BatchNorm training mode
        train_loss = 0.0
        n_batches  = 0

        for images, label_dicts in train_loader:
            images = images.to(device)

            # Forward pass: image → model → logits per field
            preds = model(images)

            # Sum cross-entropy loss across all heads (category + color)
            # Each head contributes equally. You could weight them if one matters more.
            loss = sum(
                criterion(
                    preds[field],
                    torch.tensor(
                        [label_dicts[field][i].item() for i in range(len(images))],
                        dtype=torch.long,
                        device=device,
                    )
                )
                for field in label_maps
            )

            # Backward pass + optimization step
            optimizer.zero_grad()   # clear gradients from last step
            loss.backward()         # compute new gradients via backpropagation

            # Gradient clipping: prevents explosion during backbone fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()        # update weights using gradients

            train_loss += loss.item()
            n_batches  += 1

        # ── Validation pass ────────────────────────────────────────────────────
        model.eval()    # disables Dropout, uses running stats for BatchNorm
        val_loss   = 0.0
        correct    = {field: 0 for field in label_maps}
        total      = 0

        with torch.no_grad():   # don't compute gradients — saves memory and time
            for images, label_dicts in val_loader:
                images = images.to(device)
                preds  = model(images)
                n      = len(images)

                for field in label_maps:
                    targets = torch.tensor(
                        [label_dicts[field][i].item() for i in range(n)],
                        dtype=torch.long,
                        device=device,
                    )
                    val_loss           += criterion(preds[field], targets).item()
                    # argmax: which class has the highest logit?
                    correct[field]     += (preds[field].argmax(1) == targets).sum().item()
                total += n

        avg_train = train_loss / max(n_batches, 1)
        avg_val   = val_loss   / max(len(val_loader), 1)
        accs      = {f: correct[f] / max(total, 1) for f in label_maps}
        elapsed   = time.time() - epoch_start

        scheduler.step(avg_val)

        print(f"  {epoch+1:5d} |   {phase}   | {avg_train:10.4f} | {avg_val:8.4f} | "
              f"{accs['category']:7.1%} | {accs['color']:7.1%} | {elapsed:.0f}s")

        # Save the best model checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "label_maps":  label_maps,
                "epoch":       epoch + 1,
                "val_loss":    avg_val,
                "category_acc": accs["category"],
                "color_acc":    accs["color"],
            }, MODEL_PATH)
            print(f"         ✓ Saved best model (val_loss={avg_val:.4f})")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n  Training complete.")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Model saved to:      {MODEL_PATH}")
    print(f"  Label maps saved to: {LABELS_PATH}")
    print()
    print("  Next step: restart your FastAPI server. It will auto-load the trained model.")
    print("  Run: python learn/06_evaluate.py   (to evaluate the model more thoroughly)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP for fashion classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",       type=int,   default=15,
                        help="Total training epochs (phase 1: first half, phase 2: second half)")
    parser.add_argument("--batch-size",   type=int,   default=32,
                        help="Images per batch (reduce to 16 if running out of memory)")
    parser.add_argument("--lr-head",      type=float, default=1e-3,
                        help="Learning rate for classification heads (new layers)")
    parser.add_argument("--lr-backbone",  type=float, default=1e-5,
                        help="Learning rate for CLIP backbone (phase 2, keep 100× smaller than lr-head)")
    parser.add_argument("--skip-chapters", action="store_true",
                        help="Skip educational chapters and go straight to training")
    args = parser.parse_args()

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  LESSON 5: Fine-Tuning CLIP for Fashion Classification")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Epochs:        {args.epochs}  (phase 1: first {args.epochs//2}, phase 2: last {args.epochs - args.epochs//2})")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  LR (heads):    {args.lr_head:.0e}")
    print(f"  LR (backbone): {args.lr_backbone:.0e}  ({int(args.lr_head/args.lr_backbone)}× smaller)")
    print()

    if not args.skip_chapters:
        print_chapter_1()
        input("  Press Enter to continue...")

        print_chapter_2()
        input("  Press Enter to continue...")

        print_chapter_3()
        input("  Press Enter to continue...")

        print_chapter_4()
        input("  Press Enter to continue...")

        print_chapter_5(CSV_PATH, CATEGORY_MAP, COLOR_MAP)
        input("  Press Enter to continue...")

        print_chapter_6()
        input("  Press Enter to continue...")

        print_chapter_7()
        input("  Press Enter to start training...")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  TRAINING BEGINS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    train(args.epochs, args.batch_size, args.lr_head, args.lr_backbone)

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Lesson 5 complete.")
    print("  What you learned:")
    print("    ✓ Transfer learning = borrow 86M params, add 50K of your own")
    print("    ✓ Classification heads = small translator layers on top of CLIP")
    print("    ✓ Dropout = random neuron silencing to prevent overfitting")
    print("    ✓ Two-phase training = stabilize heads first, then fine-tune backbone")
    print("    ✓ Gradient clipping = prevent explosion when training large models")
    print("    ✓ val loss is the number that matters for real-world accuracy")
    print()
    print("  Next: python learn/06_evaluate.py")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
