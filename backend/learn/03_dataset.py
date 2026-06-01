"""
LESSON 3: Datasets and DataLoaders — How to Feed Data to a Model

In lesson 2 we hardcoded X and Y in memory. That worked for 200 points.
But your fashion dataset has 44,000 images. Loading them all at once would
crash your computer. This lesson explains the solution.

Run: python learn/03_dataset.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import random
import math

torch.manual_seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1: Why Can't We Just Load All Images Into Memory?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Let's talk about actual numbers.

Your Myntra fashion dataset:
  - 44,000 images
  - Each image: ~224 × 224 pixels, 3 color channels (RGB)
  - Each pixel value: stored as float32 = 4 bytes

Memory per image (as a tensor):
  224 × 224 × 3 × 4 bytes = 602,112 bytes ≈ 0.57 MB
""")

img_size = 224
channels = 3
bytes_per_float = 4
bytes_per_image = img_size * img_size * channels * bytes_per_float
mb_per_image = bytes_per_image / (1024 * 1024)

total_images = 44_000
total_gb = (bytes_per_image * total_images) / (1024 ** 3)

print(f"  One image  (224×224×3×4):   {bytes_per_image:,} bytes  =  {mb_per_image:.2f} MB")
print(f"  44,000 images:              {bytes_per_image * total_images:,} bytes  =  {total_gb:.1f} GB")
print(f"""
  A typical laptop has 8-16 GB of total RAM.
  You'd blow the entire RAM just loading the images — before even running
  the model (which also needs memory).

  The SOLUTION: don't load everything at once.
  Load ONE image at a time, process it, then throw it away.

  This is what PyTorch's Dataset class does.

  Think of it like a library:
    - The library doesn't carry every book to your desk at once.
    - You give the librarian an index number ("book 42 please"),
      and they fetch exactly that one book from the shelf.
    - Dataset works the same way: dataset[42] → one image, one label.
""")

# Demonstrate the memory difference concretely
print("  Memory demonstration:")
print()

# Simulate 10 images loaded all at once
n_small = 10
fake_batch = torch.zeros(n_small, channels, img_size, img_size)
batch_mb = fake_batch.element_size() * fake_batch.nelement() / (1024 * 1024)
print(f"  Loading {n_small} images at once:  tensor shape {list(fake_batch.shape)}")
print(f"  Memory used:  {batch_mb:.1f} MB  (fits fine)")
print()

# Simulate what 44k would be
simulated_full_mb = mb_per_image * total_images
print(f"  Loading all {total_images:,} images at once:  would need {simulated_full_mb:.0f} MB = {total_gb:.1f} GB")
print(f"  Result: MemoryError or system freeze")
del fake_batch  # free the memory

input("\n  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2: Building a Dataset Class From Scratch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A PyTorch Dataset is just a Python class with exactly TWO required methods:

    __len__()         →  "How many total samples are there?"
    __getitem__(i)    →  "Give me sample number i"

That's it. Everything else is optional.

WHY these two methods?
  PyTorch's DataLoader (which we'll see later) needs to know the size
  so it can shuffle and batch correctly. And it fetches samples by index.

Think of __getitem__ like looking up a word in a dictionary —
  dataset[42] → the 42nd (image, label) pair, loaded on demand.

We'll build a synthetic fashion dataset: solid-color squares where:
  Red square   → "top"
  Blue square  → "jeans"
  Green square → "jacket"

This generates fake data that behaves IDENTICALLY to loading real .jpg files.
""")


class SyntheticFashionDataset(Dataset):
    """
    Generates solid-colored squares as fake fashion images.

    Color → Label mapping:
      Red-ish   → top    (label 2)
      Blue-ish  → jeans  (label 1)
      Green-ish → jacket (label 0)

    In train.py, __getitem__ opens a real .jpg from disk instead of generating
    a square — but the structure (and both required methods) are identical.
    """

    # Label map: name → integer index (must be consistent everywhere)
    LABEL_MAP = {"jacket": 0, "jeans": 1, "top": 2}

    # Base colors for each category
    BASE_COLORS = {
        "top":    (220, 80,  80),   # red-ish
        "jeans":  (60,  100, 200),  # blue-ish
        "jacket": (60,  160, 80),   # green-ish
    }

    def __init__(self, n_samples=300, img_size=64, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.samples = []  # list of (PIL Image, label_int) pairs

        categories = list(self.LABEL_MAP.keys())  # ["jacket", "jeans", "top"]

        for i in range(n_samples):
            # Pick a random category for this sample
            category = categories[i % len(categories)]  # even split across classes

            # Get base color and add small random noise so images differ
            r_base, g_base, b_base = self.BASE_COLORS[category]
            noise = 25  # ± 25 per channel
            r = max(0, min(255, r_base + random.randint(-noise, noise)))
            g = max(0, min(255, g_base + random.randint(-noise, noise)))
            b = max(0, min(255, b_base + random.randint(-noise, noise)))

            # Create the image (solid color square)
            img = Image.new("RGB", (img_size, img_size), color=(r, g, b))

            # Add a small rectangle to give the model something to learn from position
            draw = ImageDraw.Draw(img)
            offset = i % 10
            draw.rectangle([10 + offset, 10, 40 + offset, 50], fill=(r//2, g//2, b//2))

            label = self.LABEL_MAP[category]
            self.samples.append((img, label))

    def __len__(self):
        # DataLoader calls this to know how many batches to create
        return len(self.samples)

    def __getitem__(self, idx):
        # DataLoader calls this to get one (image, label) pair by index
        img, label = self.samples[idx]

        if self.transform:
            img = self.transform(img)  # convert PIL → tensor, normalize, etc.

        return img, label


# Create the dataset WITHOUT transforms first so we can inspect raw images
raw_dataset = SyntheticFashionDataset(n_samples=300, img_size=64, transform=None)

print(f"  Dataset created.")
print(f"  len(dataset) = {len(raw_dataset)}   (DataLoader will use this)")
print()
print(f"  Fetching individual samples:")
print()

category_names = ["jacket", "jeans", "top"]

for idx in [0, 1, 42, 99, 299]:
    img, label = raw_dataset[idx]
    r, g, b = img.getpixel((5, 5))  # read top-left pixel color
    print(f"  dataset[{idx:3d}]  →  label={label} ({category_names[label]:8s})  "
          f"  image size={img.size}  pixel[5,5]=RGB({r},{g},{b})")

print(f"""
  Notice:
    - dataset[0]   and dataset[3]  may share the same category (every 3rd repeats)
    - Each image has slightly different colors (random noise was added)
    - The label is an integer, NOT the string "jacket" — more on that in Chapter 3
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3: What Is a Label? What Is a Label Map?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A label is the "right answer" for each training sample.
For image classification: the label tells the model what category the image belongs to.

THE PROBLEM: models speak numbers, not words.

When you compute CrossEntropyLoss, you write:
    loss = criterion(predictions, labels)

CrossEntropyLoss expects "labels" to be integers (0, 1, 2, ...).
It cannot accept strings like "jacket" or "jeans".

THE SOLUTION: a label map.
""")

label_map = {"jacket": 0, "jeans": 1, "top": 2}
reverse_map = {v: k for k, v in label_map.items()}

print(f"  Label map (name → integer):")
for name, idx in label_map.items():
    print(f"    \"{name}\"  →  {idx}")

print(f"""
  This mapping must be CONSISTENT throughout the entire project:
    - When building the dataset:   label = label_map["jeans"]  = 1
    - When computing loss:         CrossEntropyLoss(..., tensor([1]))
    - When reading predictions:    predicted_class = 1  →  "jeans"

  If you mix up the mapping at any stage, your model will learn the wrong thing.
  A common bug: training with one label order, evaluating with a different order.

  Reverse map (integer → name)  for converting predictions back to words:
""")

print(f"  Reverse map (integer → name):")
for idx, name in reverse_map.items():
    print(f"    {idx}  →  \"{name}\"")

print(f"""
  Example prediction pipeline:
    model outputs:       [0.1, 0.7, 0.2]   (scores for jacket, jeans, top)
    argmax:              1                  (highest score is index 1)
    reverse_map[1]:      "jeans"            ← human-readable result

  Let's simulate this:
""")

# Simulate a batch of model outputs
fake_logits = torch.tensor([
    [2.1, -0.5, 0.3],   # high score for jacket (idx 0)
    [-0.2, 1.8, 0.4],   # high score for jeans  (idx 1)
    [0.1, -0.3, 2.9],   # high score for top    (idx 2)
    [-0.1, 1.2, 0.7],   # high score for jeans  (idx 1)
])
true_labels = torch.tensor([0, 1, 2, 1])

probs = torch.softmax(fake_logits, dim=1)
preds = fake_logits.argmax(dim=1)

print(f"  {'True Label':>14}  {'Predicted':>10}  {'Correct?':>10}  {'Confidence'}")
print(f"  {'----------':>14}  {'---------':>10}  {'--------':>10}  {'----------'}")
for i in range(4):
    true_name = reverse_map[true_labels[i].item()]
    pred_name = reverse_map[preds[i].item()]
    correct = "YES" if preds[i] == true_labels[i] else "NO"
    conf = probs[i, preds[i]].item()
    print(f"  {true_name:>14}  {pred_name:>10}  {correct:>10}  {conf:.1%}")

input("\n  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4: Train/Validation Split — Why It Matters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analogy: Studying for an exam vs taking a practice test.

You study from a textbook (training data).
Your professor tests you with NEW questions you've never seen (test data).

If the professor tested you with the exact same questions from the textbook,
they'd be measuring memorization, not understanding.

OVERFITTING is when a model "memorizes" training examples instead of learning
general patterns. It scores 98% on training data but only 55% on new images.

To detect overfitting, you hold out a portion of your data:
  - Training set   (80%): the model LEARNS from this
  - Validation set (20%): you CHECK performance on this — model never trains on it

Watch what happens if you test on the same data you trained on:
  - Loss goes to zero on training data (model memorized it!)
  - Loss stays high on new images (it didn't generalize)
""")

# Demonstrate overfitting signal
full_dataset = SyntheticFashionDataset(n_samples=300, img_size=32, transform=T.Compose([
    T.Resize(32),
    T.ToTensor(),
]))

val_size   = int(len(full_dataset) * 0.20)  # 20% = 60 samples
train_size = len(full_dataset) - val_size   # 80% = 240 samples

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  Total dataset:        {len(full_dataset)} samples")
print(f"  Training split:       {len(train_dataset)} samples  (80%)")
print(f"  Validation split:     {len(val_dataset)} samples  (20%)")
print(f"""
  Why 80/20?
    Common convention. Enough training data to learn patterns,
    enough validation data to get reliable accuracy estimates.
    Some projects use 70/15/15 (train/val/test) for a third "never seen" split.

  How random_split works:
    It randomly selects indices from the full dataset.
    With seed=42, it's reproducible — you get the same split every run.
    Without a seed, the split would be different each run (confusing for debugging).

  IMPORTANT BUG TO AVOID:
    After random_split, BOTH datasets still point to the SAME underlying dataset object.
    If you apply training augmentations to the full_dataset, the val split also gets augmented.
    We'll handle this properly in Chapter 5.
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5: Data Augmentation — Making Your Dataset Bigger for Free
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have 35,000 training images. Wouldn't 350,000 be better?

Data augmentation creates VARIATIONS of each image:
  flip it horizontally  →  looks like a new image to the model
  shift the brightness  →  looks like taken in different lighting
  zoom in slightly      →  looks like a different photo angle

The model sees these as distinct examples. For free.

WHY does this help?
  Overfitting happens when the model memorizes specific images.
  Augmentation means the model can never see exactly the same image twice —
  so it's forced to learn the PATTERN (what a jacket looks like), not the exact pixels.

IMPORTANT RULE:
  Apply augmentations to TRAINING data only.
  Validation/test data MUST NOT be augmented.

  Reason: you're evaluating how well the model handles real-world images.
  Augmenting validation would give you inconsistent numbers that mean nothing.
""")

# Create one sample image and show it through different augmentations
sample_img, sample_label = full_dataset[0]
sample_name = category_names[sample_label]

print(f"  Original image: label={sample_label} ({sample_name})")

# Define training transform pipeline
train_transform = T.Compose([
    T.Resize(64),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation transform: NO random ops, just resize + normalize
val_transform = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Get the PIL image from the dataset (before any transform)
raw_img = full_dataset.samples[train_dataset.indices[0]][0]

print(f"\n  Applying training augmentations 5 times to the same image:")
print(f"  (Each run applies random ops, so results differ)\n")
print(f"  {'Run':>5}  {'Tensor shape':>15}  {'Min val':>10}  {'Max val':>10}  {'Mean val':>10}")
print(f"  {'---':>5}  {'------------':>15}  {'-------':>10}  {'-------':>10}  {'--------':>10}")

for run in range(1, 6):
    tensor = train_transform(raw_img)
    print(f"  {run:>5}  {str(list(tensor.shape)):>15}  {tensor.min().item():>10.4f}  "
          f"{tensor.max().item():>10.4f}  {tensor.mean().item():>10.4f}")

print(f"""
  Notice: same image, different min/max/mean each run.
  That's the augmentation working — each run is slightly different.

  Now the VALIDATION transform (no random ops) on the same image:
""")

print(f"  {'Run':>5}  {'Tensor shape':>15}  {'Min val':>10}  {'Max val':>10}  {'Mean val':>10}")
print(f"  {'---':>5}  {'------------':>15}  {'-------':>10}  {'-------':>10}  {'--------':>10}")

for run in range(1, 4):
    tensor = val_transform(raw_img)
    print(f"  {run:>5}  {str(list(tensor.shape)):>15}  {tensor.min().item():>10.4f}  "
          f"{tensor.max().item():>10.4f}  {tensor.mean().item():>10.4f}")

print(f"""
  Validation transform is IDENTICAL every run — consistent, reproducible.

  The Normalize transform:
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    This shifts pixel values from [0, 1] to [-1, 1].
    Formula per channel:  normalized = (original - mean) / std
                                     = (original - 0.5) / 0.5

    Why? Neural networks learn faster when inputs are centered around 0.
    CLIP uses ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
""")

# Rebuild the datasets with proper transforms applied to separate subsets
train_data_aug = SyntheticFashionDataset(n_samples=240, img_size=64, transform=train_transform)
val_data_aug   = SyntheticFashionDataset(n_samples=60,  img_size=64, transform=val_transform)

print(f"  Rebuilt datasets with correct transform assignment:")
print(f"    train_dataset: {len(train_data_aug)} samples, training transforms (augmented)")
print(f"    val_dataset:   {len(val_data_aug)} samples, val transforms (no augmentation)")

input("\n  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 6: DataLoader — The Engine That Feeds Batches to Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset handles one image at a time.
DataLoader handles BATCHES — many images grouped together.

WHY batches instead of one image at a time?

  Option A — one image per update:
    Very slow (a GPU is designed to process many things in parallel, not one).
    Noisy — each single image's gradient could be misleading.

  Option B — all 44,000 images at once:
    Crashes your RAM (we showed this in Chapter 1).

  Option C — batches of 32 (or 64, or 128):
    GPU processes 32 images simultaneously → fast.
    Gradient is averaged across 32 images → smoother updates.
    Memory stays manageable.

The tradeoff:
  Larger batch:  faster per epoch, uses more GPU memory, may overfit
  Smaller batch: slower per epoch, less memory, better regularization

Common batch sizes: 16, 32, 64, 128  (always powers of 2 for GPU efficiency)
""")

train_loader = DataLoader(
    train_data_aug,
    batch_size=32,
    shuffle=True,     # IMPORTANT: shuffle training data each epoch
    num_workers=0,    # workers = parallel loading processes. 0 = main thread (safe on Mac)
    drop_last=False,  # keep last batch even if it's smaller than batch_size
)

val_loader = DataLoader(
    val_data_aug,
    batch_size=32,
    shuffle=False,    # NEVER shuffle validation — results must be reproducible
    num_workers=0,
)

print(f"  Training DataLoader:")
print(f"    dataset size:   {len(train_data_aug)}")
print(f"    batch_size:     32")
print(f"    batches/epoch:  {len(train_loader)}   (= ceil({len(train_data_aug)} / 32))")
print(f"    shuffle:        True")
print()
print(f"  Validation DataLoader:")
print(f"    dataset size:   {len(val_data_aug)}")
print(f"    batch_size:     32")
print(f"    batches/epoch:  {len(val_loader)}")
print(f"    shuffle:        False")
print()
print(f"  Why shuffle=True for training?")
print(f"    If images always come in the same order (all jackets first, then jeans...)")
print(f"    the model learns the ORDER, not the features. Shuffling prevents this.")
print()

# Inspect one real batch
images_batch, labels_batch = next(iter(train_loader))

print(f"  One training batch:")
print(f"    images shape:  {list(images_batch.shape)}")
print(f"    labels shape:  {list(labels_batch.shape)}")
print()
print(f"  Shape breakdown: [batch, channels, height, width]")
print(f"    batch=32: 32 images in this batch")
print(f"    channels=3: RGB (red, green, blue)")
print(f"    height=64, width=64: image dimensions")
print()

# Show label distribution
label_counts = {0: 0, 1: 0, 2: 0}
for l in labels_batch.tolist():
    label_counts[l] += 1

print(f"  Label distribution in this batch:")
for idx, name in reverse_map.items():
    count = label_counts[idx]
    bar = "█" * count
    print(f"    {idx} ({name:8s}):  {count:2d}  {bar}")

print(f"""
  The label counts are roughly equal because we built a balanced dataset.
  In the real Myntra dataset, some categories are more common than others —
  that imbalance can hurt training (model learns to always predict the common class).

  When you loop through the entire DataLoader, you get every sample exactly once:
""")

# Show what iterating looks like
print(f"  Iterating over train_loader:")
total_seen = 0
for batch_idx, (imgs, lbls) in enumerate(train_loader):
    total_seen += len(imgs)
    if batch_idx < 3 or batch_idx == len(train_loader) - 1:
        print(f"    Batch {batch_idx+1:2d}: got {len(imgs)} images  (running total: {total_seen})")
    elif batch_idx == 3:
        print(f"    ...")
print(f"  Total images seen: {total_seen}  (= full dataset, exactly once = 1 epoch)")

input("\n  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 7: How Your Real Myntra Dataset Maps Onto This
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Everything we built here is what train.py uses.
The only difference: instead of generating colored squares,
the real dataset reads .jpg files from disk.

Let's see the parallel:
""")

print("""  SYNTHETIC DATASET (this file)          REAL DATASET (train.py)
  ─────────────────────────────────      ──────────────────────────────────────
  generates a color square           →   Image.open(row["image_path"])
  LABEL_MAP = {"jacket":0,...}       →   label map built from styles.csv
  random.choice(categories)         →   row["articleType"] from styles.csv
  PIL Image created in memory        →   PIL Image loaded from disk
  transform applied in __getitem__   →   same CLIP transform pipeline
  __len__ returns n_samples          →   __len__ returns len(self.df)
  __getitem__(idx) generates square  →   __getitem__(idx) reads .jpg file
""")

print("""  What styles.csv looks like (first few rows):

    id        ,articleType  ,gender  ,baseColour  ,season  ,...
    15970     ,Jeans        ,Men     ,Blue        ,Summer  ,...
    39386     ,Shirts       ,Men     ,Gray        ,Winter  ,...
    59263     ,Tshirts      ,Men     ,Gray        ,Summer  ,...
    21379     ,Watches      ,Unisex  ,Silver      ,Winter  ,...

  train.py reads "articleType" column and converts it to a label integer.
  The label map is built from the unique values in that column:

    articleType → label int
    ─────────────────────────
    "Jeans"     → 0
    "Shirts"    → 1
    "Tshirts"   → 2
    "Watches"   → 3
    ...         → ...
""")

print("""  The DataLoader structure in train.py is also identical:

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,   # from config (default 32)
        shuffle=True,
        num_workers=cfg.num_workers, # from config (default 4)
        pin_memory=True,             # speeds up CPU→GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,               # never shuffle validation
        num_workers=cfg.num_workers,
    )

  pin_memory=True: pre-loads batches into page-locked RAM for faster GPU transfer.
  You didn't see it above because we're running on CPU.
""")

# Quick end-to-end demo: a real training run on the synthetic dataset
print("""  Quick demo — a real training run using our synthetic dataset:
  (This uses the same 4-step loop from Lesson 2)
""")

class TinyCNN(nn.Module):
    """Mini CNN — same structure as what CLIP uses, just much smaller."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3ch → 16ch feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64×64 → 32×32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 16ch → 32ch feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32×32 → 16×16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),    # randomly zero 30% of activations — prevents memorization
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model     = TinyCNN(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()
print(f"  {'Epoch':>7}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>9}  {'Notes'}")
print(f"  {'-----':>7}  {'----------':>12}  {'--------':>10}  {'-------':>9}  {'-----'}")

for epoch in range(12):

    # ── Training phase ────────────────────────────────────────────────────────
    model.train()       # IMPORTANT: enables dropout for training
    running_train_loss = 0.0

    for images, labels in train_loader:
        # The 4 steps from Lesson 2:
        preds = model(images)                    # Step 1: forward pass
        loss  = criterion(preds, labels)         # Step 2: compute loss
        optimizer.zero_grad()                    # Step 3a: clear old gradients
        loss.backward()                          # Step 3b: compute new gradients
        optimizer.step()                         # Step 4: update all weights
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)

    # ── Validation phase ──────────────────────────────────────────────────────
    model.eval()        # IMPORTANT: disables dropout for consistent evaluation
    running_val_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():   # saves memory: no gradient tracking needed here
        for images, labels in val_loader:
            preds = model(images)
            running_val_loss += criterion(preds, labels).item()
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total   += len(labels)

    avg_val_loss = running_val_loss / len(val_loader)
    val_acc      = correct / total

    notes = ""
    if epoch == 0:   notes = "← initial (random weights)"
    if epoch == 5:   notes = "← improving"
    if epoch == 11:  notes = "← converged"

    print(f"  {epoch+1:>7}  {avg_train_loss:>12.4f}  {avg_val_loss:>10.4f}  {val_acc:>9.1%}  {notes}")

print(f"""
  Watch train loss vs val loss:
    If both go down together → model is learning well
    If train loss drops but val loss stays high → OVERFITTING
    If neither drops → learning rate may be wrong, or model too simple
""")

input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU LEARNED:

  Memory problem     → 44,000 images × 0.57 MB = ~25 GB. Can't load all at once.
  Dataset class      → Loads ONE image at a time on demand via __getitem__(i)
  Required methods   → __len__() and __getitem__(i). That's all you must implement.
  Labels             → Integers, not strings. label_map converts name → integer.
  Reverse map        → Converts model output integer back to human-readable name.
  Train/val split    → 80% for learning, 20% for honest evaluation. Never mix them.
  Overfitting        → Model memorizes training data, fails on new images.
                       Detected by: val loss stops improving while train loss drops.
  Augmentation       → Random flips/crops/color shifts on TRAINING data only.
                       Validation data must NEVER be augmented.
  DataLoader         → Groups samples into batches. Handles shuffling and workers.
  shuffle=True       → Only for training. Prevents learning data order by accident.
  model.train()      → Enables dropout. Call before training loop.
  model.eval()       → Disables dropout. Call before validation/inference.
  torch.no_grad()    → Skips gradient tracking. Use during validation to save memory.

CHECKLIST FOR EVERY TRAINING RUN:
  [ ] Training data gets augmentation transforms
  [ ] Validation data gets clean (no augmentation) transforms
  [ ] train_loader has shuffle=True
  [ ] val_loader has shuffle=False
  [ ] Call model.train() before training batches
  [ ] Call model.eval() before validation batches
  [ ] Wrap validation in torch.no_grad()

Next: learn/04_clip_zero_shot.py  →  How CLIP understands fashion images
""")

print("  Lesson 3 complete.  Run: python learn/04_clip_zero_shot.py")
