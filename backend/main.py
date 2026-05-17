from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import json
import io
from pathlib import Path

from database import init_db
from shein_api import search_shein_products

FINETUNED_MODEL_PATH = Path("data/shein_model.pt")
LABEL_MAPS_PATH      = Path("data/label_maps.json")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load fine-tuned model if available (created by train.py)
# ft_model loaded below after all class/function definitions

COLOR_LABELS = [
    "white clothing", "black clothing", "blue clothing", "red clothing",
    "pink clothing", "green clothing", "beige or cream clothing", "brown clothing",
    "gray clothing", "yellow clothing", "purple clothing", "orange clothing",
    "multicolor or printed clothing",
]

CATEGORY_LABELS = [
    "a t-shirt or casual top", "a blouse or dressy top", "a sweater or hoodie",
    "a jacket, coat, or blazer", "jeans or denim pants", "trousers or dress pants",
    "shorts", "a dress", "a skirt", "sneakers or athletic shoes", "boots", "sandals or heels",
]

STYLE_LABELS = [
    "casual outfit", "formal outfit", "business casual outfit", "streetwear outfit",
    "athleisure outfit", "minimal outfit", "summer outfit", "party outfit",
]

# Per-category style attribute groups — each group is one CLIP pass, top result wins
STYLE_ATTRIBUTES = {
    "jeans": [
        {
            "labels": ["wide leg jeans", "straight leg jeans", "skinny jeans", "slim fit jeans", "baggy jeans", "flared jeans", "bootcut jeans"],
            "strip": " jeans",
        },
        {
            "labels": ["high waist jeans", "mid rise jeans", "low rise jeans"],
            "strip": " jeans",
        },
        {
            "labels": ["distressed ripped jeans", "plain solid jeans", "embroidered jeans", "patchwork jeans"],
            "strip": " jeans",
        },
    ],
    "trousers": [
        {
            "labels": ["wide leg trousers", "straight leg trousers", "slim fit trousers", "flared trousers", "pleated trousers"],
            "strip": " trousers",
        },
        {
            "labels": ["high waist trousers", "mid rise trousers"],
            "strip": " trousers",
        },
    ],
    "shorts": [
        {
            "labels": ["denim shorts", "athletic shorts", "tailored shorts", "biker shorts", "cargo shorts"],
            "strip": " shorts",
        },
        {
            "labels": ["high waist shorts", "low rise shorts"],
            "strip": " shorts",
        },
    ],
    "top": [
        # Length
        {
            "labels": ["cropped top above the waist", "standard length top at the hip", "longline top below the hip"],
            "strip": " top above the waist|top at the hip|top below the hip",
            "remap": {
                "cropped top above the waist": "cropped",
                "standard length top at the hip": "",
                "longline top below the hip": "longline",
            },
        },
        # Fit
        {
            "labels": ["oversized baggy top", "fitted bodycon top", "relaxed loose top"],
            "strip": " top",
            "remap": {
                "oversized baggy top": "oversized",
                "fitted bodycon top": "fitted",
                "relaxed loose top": "",
            },
        },
        # Neckline
        {
            "labels": [
                "v-neck top", "deep v-neck top", "square neck top", "crew neck top",
                "scoop neck top", "off-shoulder top", "one shoulder top",
                "halter neck top", "cowl neck top", "sweetheart neck top",
                "mock neck top", "tie neck top",
            ],
            "strip": " top",
        },
        # Sleeve length & style
        {
            "labels": [
                "sleeveless tank top", "short sleeve top", "long sleeve top",
                "three quarter sleeve top", "puff sleeve top", "flutter sleeve top",
            ],
            "strip": " top",
            "remap": {
                "sleeveless tank top": "sleeveless",
                "short sleeve top": "short sleeve",
                "long sleeve top": "long sleeve",
                "three quarter sleeve top": "three quarter sleeve",
                "puff sleeve top": "puff sleeve",
                "flutter sleeve top": "flutter sleeve",
            },
        },
        # Style details
        {
            "labels": [
                "ruched top", "smocked top", "corset top", "wrap top",
                "tie front top", "peplum top", "lace trim top", "cutout top",
                "ribbed top", "plain solid top",
            ],
            "strip": " top",
        },
        # Print
        {
            "labels": [
                "solid plain top", "striped top", "floral top",
                "graphic print top", "animal print top", "abstract print top",
            ],
            "strip": " top",
        },
    ],
    "blouse": [
        # Length
        {
            "labels": ["cropped blouse", "standard length blouse", "longline blouse"],
            "strip": " blouse",
        },
        # Fit & style
        {
            "labels": [
                "flowy blouse", "fitted blouse", "wrap blouse", "oversized blouse",
                "peplum blouse", "tie front blouse", "smocked blouse", "ruched blouse",
            ],
            "strip": " blouse",
        },
        # Neckline
        {
            "labels": [
                "v-neck blouse", "square neck blouse", "off-shoulder blouse",
                "one shoulder blouse", "halter neck blouse", "scoop neck blouse",
                "crew neck blouse", "tie neck blouse",
            ],
            "strip": " blouse",
        },
        # Sleeve
        {
            "labels": [
                "sleeveless blouse", "short sleeve blouse", "long sleeve blouse",
                "puff sleeve blouse", "flutter sleeve blouse",
            ],
            "strip": " blouse",
        },
        # Print
        {
            "labels": ["floral blouse", "solid blouse", "striped blouse", "printed blouse", "lace blouse"],
            "strip": " blouse",
        },
    ],
    "sweater": [
        {
            "labels": ["oversized sweater", "fitted sweater", "cropped sweater"],
            "strip": " sweater",
        },
        {
            "labels": ["crew neck sweater", "v-neck sweater", "turtleneck sweater", "cardigan sweater"],
            "strip": " sweater",
        },
        {
            "labels": ["cable knit sweater", "ribbed sweater", "plain solid sweater"],
            "strip": " sweater",
        },
    ],
    "jacket": [
        {
            "labels": ["denim jacket", "leather jacket", "blazer jacket", "bomber jacket", "puffer jacket", "trench coat"],
            "strip": " jacket",
        },
        {
            "labels": ["cropped jacket", "oversized jacket", "fitted jacket"],
            "strip": " jacket",
        },
    ],
    "dress": [
        {
            "labels": ["mini dress", "midi dress", "maxi dress"],
            "strip": " dress",
        },
        {
            "labels": ["bodycon dress", "flowy a-line dress", "wrap dress", "slip dress", "shirt dress"],
            "strip": " dress",
        },
        {
            "labels": ["sleeveless dress", "short sleeve dress", "long sleeve dress", "off-shoulder dress"],
            "strip": " dress",
        },
        {
            "labels": ["floral dress", "solid dress", "striped dress", "printed dress"],
            "strip": " dress",
        },
    ],
    "skirt": [
        {
            "labels": ["mini skirt", "midi skirt", "maxi skirt"],
            "strip": " skirt",
        },
        {
            "labels": ["pleated skirt", "straight skirt", "flared skirt", "wrap skirt", "satin skirt"],
            "strip": " skirt",
        },
    ],
    "sneakers": [
        {
            "labels": ["low top sneakers", "high top sneakers", "platform sneakers", "chunky sneakers", "slip on sneakers"],
            "strip": " sneakers",
        },
    ],
    "boots": [
        {
            "labels": ["ankle boots", "knee high boots", "combat boots", "platform boots", "cowboy boots"],
            "strip": " boots",
        },
    ],
    "sandals": [
        {
            "labels": ["flat sandals", "heeled sandals", "platform sandals", "strappy sandals", "mule sandals"],
            "strip": " sandals",
        },
    ],
}

CATEGORY_SEARCH_TERMS = {
    "a t-shirt or casual top": "top",
    "a blouse or dressy top": "blouse",
    "a sweater or hoodie": "sweater",
    "a jacket, coat, or blazer": "jacket",
    "jeans or denim pants": "jeans",
    "trousers or dress pants": "trousers",
    "shorts": "shorts",
    "a dress": "dress",
    "a skirt": "skirt",
    "sneakers or athletic shoes": "sneakers",
    "boots": "boots",
    "sandals or heels": "sandals",
}

# Slot groupings — all labels compete in ONE CLIP pass so probabilities are meaningful
OUTFIT_SLOTS = {
    "top":    ["a t-shirt or casual top", "a blouse or dressy top", "a sweater or hoodie", "a jacket, coat, or blazer"],
    "bottom": ["jeans or denim pants", "trousers or dress pants", "shorts", "a skirt"],
    "shoes":  ["sneakers or athletic shoes", "boots", "sandals or heels"],
    "dress":  ["a dress"],
}

COLORS_SHORT = ["white", "black", "blue", "red", "pink", "green", "beige", "brown", "gray", "yellow", "purple", "orange"]

COLOR_SEARCH_TERMS = {
    "white clothing": "white", "black clothing": "black", "blue clothing": "blue",
    "red clothing": "red", "pink clothing": "pink", "green clothing": "green",
    "beige or cream clothing": "beige", "brown clothing": "brown", "gray clothing": "gray",
    "yellow clothing": "yellow", "purple clothing": "purple", "orange clothing": "orange",
    "multicolor or printed clothing": "",
}


# ── Fine-tuned model (loaded when available) ─────────────────────────────────

class SheinClassifier(nn.Module):
    def __init__(self, label_maps, clip_model):
        super().__init__()
        self.clip_vision = clip_model.vision_model
        self.clip_proj   = clip_model.visual_projection
        embed_dim = 512
        self.heads = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, len(lmap)),
            )
            for field, lmap in label_maps.items()
        })

    def forward(self, pixel_values):
        vision_out = self.clip_vision(pixel_values=pixel_values)
        features   = self.clip_proj(vision_out.pooler_output)
        features   = features / features.norm(dim=-1, keepdim=True)
        return {field: head(features) for field, head in self.heads.items()}


def _load_finetuned_model():
    if not FINETUNED_MODEL_PATH.exists() or not LABEL_MAPS_PATH.exists():
        return None, None

    try:
        with open(LABEL_MAPS_PATH) as f:
            label_maps = json.load(f)

        clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ft_model  = SheinClassifier(label_maps, clip_base)
        checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location="cpu")
        ft_model.load_state_dict(checkpoint["model_state"])
        ft_model.eval()

        # Image transform matching CLIP preprocessing
        clip_size = processor.image_processor.size["shortest_edge"]
        ft_transform = transforms.Compose([
            transforms.Resize(clip_size),
            transforms.CenterCrop(clip_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_processor.image_mean,
                std=processor.image_processor.image_std,
            ),
        ])

        print("Fine-tuned Shein model loaded.")
        return ft_model, {"label_maps": label_maps, "transform": ft_transform}
    except Exception as e:
        print(f"Could not load fine-tuned model: {e}")
        return None, None


def predict_finetuned(pil_image, ft_model, ft_config):
    """Run fine-tuned model and return detected attributes."""
    label_maps = ft_config["label_maps"]
    transform  = ft_config["transform"]

    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        preds = ft_model(tensor)

    result = {}
    for field, logits in preds.items():
        idx   = logits.argmax(1).item()
        # invert label_map to get index→label
        inv   = {v: k for k, v in label_maps[field].items()}
        result[field] = inv.get(idx, "")

    return result


# ─────────────────────────────────────────────────────────────────────────────

def predict_top(image, labels, top_k=1):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    top_indices = probs.topk(top_k).indices.tolist()
    return [(labels[i], probs[i].item()) for i in top_indices]


def detect_style_attributes(image, category):
    attribute_groups = STYLE_ATTRIBUTES.get(category, [])
    descriptors = []

    for group in attribute_groups:
        top = predict_top(image, group["labels"], top_k=1)[0]
        label = top[0]

        if "remap" in group:
            descriptor = group["remap"].get(label, label)
        else:
            descriptor = label.replace(group["strip"], "").strip()

        if descriptor and descriptor not in ("plain solid", "regular fit", "solid", "standard length", "relaxed loose"):
            descriptors.append(descriptor)

    return descriptors


def detect_color_for_item(image, category):
    """Detect the color of a specific clothing item using combined color+category labels."""
    labels = [f"{c} {category}" for c in COLORS_SHORT]
    results = predict_top(image, labels, top_k=1)
    winning = results[0][0]  # e.g. "white top"
    return winning.split()[0]  # return just the color word


def detect_outfit(image):
    # ── Step 1: all category labels in one CLIP pass ──────────────────────────
    # Probabilities sum to 1 across all labels so scores are meaningful
    all_labels = CATEGORY_LABELS  # 12 labels total
    all_results = predict_top(image, all_labels, top_k=len(all_labels))
    prob_map = {label: conf for label, conf in all_results}

    # Best score per slot
    slot_best = {}
    for slot, labels in OUTFIT_SLOTS.items():
        for label in labels:
            conf = prob_map.get(label, 0)
            if slot not in slot_best or conf > slot_best[slot][1]:
                slot_best[slot] = (label, conf)

    # ── Step 2: dress vs top+bottom decision ─────────────────────────────────
    dress_conf  = slot_best.get("dress",  ("", 0))[1]
    top_conf    = slot_best.get("top",    ("", 0))[1]
    bottom_conf = slot_best.get("bottom", ("", 0))[1]

    # Dress wins only if it outscores top AND bottom combined
    if dress_conf > (top_conf + bottom_conf):
        active_slots = ["dress", "shoes"]
    else:
        active_slots = ["top", "bottom", "shoes"]

    # ── Step 3: build result per active slot ─────────────────────────────────
    items = []
    for slot in active_slots:
        if slot not in slot_best:
            continue
        best_label, confidence = slot_best[slot]

        # Require meaningful confidence — in a 12-class competition ~0.08 is random chance
        if confidence < 0.08:
            continue

        category = CATEGORY_SEARCH_TERMS.get(best_label, "")
        if not category:
            continue

        # Per-item color: always ask CLIP "white jeans" vs "blue jeans" etc.
        # Never use the fine-tuned global color here — it picks the dominant
        # color in the whole image (e.g. blue jeans) and applies it to everything.
        color = detect_color_for_item(image, category)

        style_attrs = detect_style_attributes(image, category)
        attrs_str   = " ".join(style_attrs)
        parts       = [p for p in [color, attrs_str, category] if p]
        query       = " ".join(parts)

        items.append({
            "category":        category,
            "color":           color,
            "style_attributes": style_attrs,
            "query":           query,
            "confidence":      round(confidence, 3),
        })

    return items


# Load fine-tuned model if available (created by train.py)
ft_model, ft_config = _load_finetuned_model()


@app.get("/")
def read_root():
    return {"message": "Backend is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    detected_items = detect_outfit(image)
    style = predict_top(image, STYLE_LABELS, top_k=1)[0][0]

    recommendations = {}
    for item in detected_items:
        category = item["category"]
        if category in recommendations:
            continue
        recommendations[category] = search_shein_products(item["query"], limit=5)

    return {
        "filename": file.filename,
        "detected_items": detected_items,
        "style": style,
        "recommendations": recommendations,
    }
