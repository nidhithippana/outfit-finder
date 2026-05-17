"""
Fine-tunes CLIP on the Myntra fashion dataset for category + color classification.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 32

After training, drop data/shein_model.pt + data/label_maps.json into the backend
and restart the server — it will auto-load the fine-tuned model.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DATASET_DIR = Path("/Users/nidhithippana/Desktop/DEV/outfit-finder-ai/data/myntradataset")
IMAGES_DIR  = DATASET_DIR / "images"
CSV_PATH    = DATASET_DIR / "styles.csv"

MODEL_PATH  = Path("data/shein_model.pt")
LABELS_PATH = Path("data/label_maps.json")

# ── Label mappings ────────────────────────────────────────────────────────────

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


def load_dataset():
    rows = []
    with open(CSV_PATH, errors="ignore") as f:
        for row in csv.DictReader(f):
            if row.get("gender") not in ("Women", "Girls"):
                continue
            category = CATEGORY_MAP.get(row.get("articleType", ""))
            color    = COLOR_MAP.get(row.get("baseColour", ""))
            if not category or not color:
                continue
            img_path = IMAGES_DIR / f"{row['id']}.jpg"
            if not img_path.exists():
                continue
            rows.append({
                "image_path": str(img_path),
                "category":   category,
                "color":      color,
            })

    print(f"Loaded {len(rows)} samples after filtering")
    return rows


def build_label_maps(rows):
    categories = sorted(set(r["category"] for r in rows))
    colors     = sorted(set(r["color"]    for r in rows))
    return {
        "category": {v: i for i, v in enumerate(categories)},
        "color":    {v: i for i, v in enumerate(colors)},
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

class FashionDataset(Dataset):
    def __init__(self, rows, label_maps, transform):
        self.rows       = rows
        self.label_maps = label_maps
        self.transform  = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        image = self.transform(image)
        labels = {
            "category": self.label_maps["category"][row["category"]],
            "color":    self.label_maps["color"][row["color"]],
        }
        return image, labels


# ── Model ─────────────────────────────────────────────────────────────────────

class FashionClassifier(nn.Module):
    def __init__(self, label_maps, clip_model):
        super().__init__()
        self.clip_vision = clip_model.vision_model
        self.clip_proj   = clip_model.visual_projection
        embed_dim = 512

        self.heads = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, len(lmap)),
            )
            for field, lmap in label_maps.items()
        })

    def forward(self, pixel_values):
        vision_out = self.clip_vision(pixel_values=pixel_values)
        features   = self.clip_proj(vision_out.pooler_output)
        features   = features / features.norm(dim=-1, keepdim=True)
        return {field: head(features) for field, head in self.heads.items()}


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs, batch_size, lr_head, lr_backbone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Note: Training on CPU will take ~1-2 hours. Consider running on Google Colab for GPU speed.")

    rows = load_dataset()
    if len(rows) < 50:
        print("Not enough data found. Check that DATASET_DIR path is correct.")
        return

    label_maps = build_label_maps(rows)
    Path("data").mkdir(exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"Categories: {list(label_maps['category'].keys())}")
    print(f"Colors:     {list(label_maps['color'].keys())}")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_size  = processor.image_processor.size["shortest_edge"]

    transform = transforms.Compose([
        transforms.Resize(clip_size),
        transforms.CenterCrop(clip_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_processor.image_mean,
            std=processor.image_processor.image_std,
        ),
    ])

    dataset   = FashionDataset(rows, label_maps, transform)
    val_size  = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model      = FashionClassifier(label_maps, clip_model).to(device)

    # Phase 1: freeze CLIP, train heads only
    for p in model.clip_vision.parameters(): p.requires_grad = False
    for p in model.clip_proj.parameters():   p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=lr_head)
    best_val  = float("inf")

    for epoch in range(epochs):
        # Phase 2: unfreeze backbone halfway through
        if epoch == epochs // 2:
            print("\nUnfreezing CLIP backbone...")
            for p in model.clip_vision.parameters(): p.requires_grad = True
            for p in model.clip_proj.parameters():   p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.clip_vision.parameters()) + list(model.clip_proj.parameters()),
                "lr": lr_backbone,
            })

        model.train()
        train_loss = 0.0
        for images, label_dict in train_loader:
            images = images.to(device)
            preds  = model(images)
            loss   = sum(
                criterion(preds[f], torch.tensor([label_dict[f][i].item() for i in range(len(images))], device=device))
                for f in label_maps
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct  = {f: 0 for f in label_maps}
        total    = 0
        with torch.no_grad():
            for images, label_dict in val_loader:
                images = images.to(device)
                preds  = model(images)
                for f in label_maps:
                    targets = torch.tensor([label_dict[f][i].item() for i in range(len(images))], device=device)
                    val_loss += criterion(preds[f], targets).item()
                    correct[f] += (preds[f].argmax(1) == targets).sum().item()
                total += len(images)

        avg_t = train_loss / len(train_loader)
        avg_v = val_loss   / len(val_loader)
        accs  = {f: f"{correct[f]/total:.0%}" for f in label_maps}
        print(f"Epoch {epoch+1:2d}/{epochs} | train {avg_t:.3f} | val {avg_v:.3f} | {accs}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({"model_state": model.state_dict(), "label_maps": label_maps}, MODEL_PATH)
            print(f"  ✓ Saved to {MODEL_PATH}")

    print(f"\nDone. Best val loss: {best_val:.3f}")
    print(f"Model: {MODEL_PATH}  |  Labels: {LABELS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr-head",     type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr_head, args.lr_backbone)
