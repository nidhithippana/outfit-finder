"""
LESSON 4: How CLIP Works — and What main.py Is Actually Doing

Run: python learn/04_clip_zero_shot.py
(First run downloads ~600MB CLIP model — then cached forever.)

By the end of this lesson you will understand:
  - What an "embedding" is and why it exists
  - How CLIP learned to match images with text
  - What cosine similarity measures
  - How zero-shot classification works step by step
  - Why main.py is slow and what fine-tuning fixes

No machine learning background needed. We start from zero.
"""

import math
import time

# ─── Try loading CLIP. If it fails, we teach with toy vectors instead. ────────
CLIP_AVAILABLE = False
try:
    import torch
    from PIL import Image, ImageDraw
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP... (first run downloads ~600MB, then cached forever)")
    _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    print("CLIP loaded successfully.\n")
    CLIP_AVAILABLE = True
except Exception as e:
    print(f"CLIP not available ({e}). Concepts will be shown with toy vectors.\n")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1: What Is an Embedding?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before we touch any AI model, we need to understand ONE core idea:
everything a neural network works with is a LIST OF NUMBERS.

A photo. A sentence. A word. A sound clip. It doesn't matter.
The model converts it all into a list of numbers called an EMBEDDING.

  "blue jeans"  →  [0.12, -0.87, 0.34, 0.05, ...]   (512 numbers)
  [photo of blue jeans]  →  [0.11, -0.84, 0.38, 0.06, ...]   (512 numbers)

WHY does this matter? Because now you can do MATH on meaning.

  ANALOGY: GPS coordinates.
  ─────────────────────────
  Every city on Earth can be described by just TWO numbers: latitude, longitude.
    New York  = [40.7, -74.0]
    Boston    = [42.3, -71.1]
    Los Angeles = [34.0, -118.2]

  If two cities have similar coordinates → they are physically close.
  Simple math tells you which cities are "neighbors."

  Embeddings do the SAME THING — but for meaning, not geography.
  And instead of 2 dimensions, they use 512.

  "blue jeans"  lives near  "denim pants"   in 512-d space.
  "blue jeans"  lives FAR from  "pizza"     in 512-d space.

WHY 512 dimensions and not just 2?

  With only 2 dimensions (like GPS), you can only capture 2 aspects of a thing.
  With 512, you can capture 512 DIFFERENT ASPECTS simultaneously:
    dimension 1 might represent "how formal is this?"
    dimension 2 might represent "what color family?"
    dimension 3 might represent "is this a top or a bottom?"
    ... and so on for 512 different aspects.

  More dimensions = more nuance = better at distinguishing similar things.
  (512 is a design choice — CLIP's designers found it works well.)
""")
input("  Press Enter to continue...")

# ── Show the "similar things have similar numbers" idea with toy vectors ───────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1 DEMO: Similar sentences → similar numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

if CLIP_AVAILABLE:
    # Encode a few sentences and compute distances
    test_sentences = [
        "blue jeans",
        "denim pants",
        "red dress",
        "evening gown",
        "pizza",
    ]

    inputs = _clip_processor(text=test_sentences, return_tensors="pt", padding=True)
    with torch.no_grad():
        out    = _clip_model.text_model(**{k: v for k, v in inputs.items()
                                           if k in ("input_ids", "attention_mask")})
        embeds = _clip_model.text_projection(out.pooler_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)  # normalize to length 1

    print("Comparing 'blue jeans' to other phrases (1.0 = identical, 0.0 = unrelated):\n")
    ref = embeds[0]
    for i, phrase in enumerate(test_sentences):
        sim = (ref @ embeds[i]).item()
        bar = "█" * int(sim * 40)
        print(f"  'blue jeans' vs '{phrase}': {sim:.3f}  {bar}")

    print()
    print("  → 'denim pants' is very similar to 'blue jeans' (different words, same meaning!)")
    print("  → 'pizza' is almost completely unrelated (different world entirely)")
    print("  → 'red dress' and 'evening gown' cluster near each other too")
    print()
    print("This is the magic of embeddings: similar MEANING → similar NUMBERS.")

else:
    # Toy demo without CLIP
    print("Toy demo (CLIP unavailable — using manually crafted 3-d vectors):\n")
    toy_vectors = {
        "blue jeans":  [0.9, 0.1, 0.0],
        "denim pants": [0.85, 0.15, 0.0],
        "red dress":   [0.0, 0.9, 0.1],
        "pizza":       [0.0, 0.0, 1.0],
    }

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def norm(v):
        return math.sqrt(sum(x**2 for x in v))

    def cosine(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    print("Comparing 'blue jeans' to other phrases:\n")
    ref = toy_vectors["blue jeans"]
    for phrase, vec in toy_vectors.items():
        sim = cosine(ref, vec)
        bar = "█" * int(sim * 30)
        print(f"  'blue jeans' vs '{phrase}': {sim:.3f}  {bar}")

print()
input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2: What CLIP Learned from 400 Million Images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OpenAI collected 400 MILLION image-text pairs from the internet:

  [photo of blue jeans on a model]  +  caption: "blue jeans"
  [product shot of red dress]       +  caption: "red cocktail dress"
  [street photo of white sneakers]  +  caption: "white sneakers outfit"
  ... × 400,000,000

CLIP has TWO encoders that it trained simultaneously:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   Image  ──→  [Image Encoder / ViT] ──→  512-d vector   │
  │                                                          │
  │   Text   ──→  [Text Encoder]        ──→  512-d vector   │
  │                                                          │
  │   Goal: matching pairs should land NEAR each other       │
  │         non-matching pairs should land FAR apart         │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

The training objective (called CONTRASTIVE LEARNING):

  Given a batch of 32 image-text pairs:
    - 32 "positive" pairs (image 1 goes with text 1, etc.)
    - 32×32 - 32 = 992 "negative" pairs (image 1 does NOT go with text 2, 3, ...)

  Reward: push positive pairs closer → high similarity score
  Punish: push negative pairs apart  → low similarity score

  After 400M examples: the model learns that
    "any photo of jeans" lives in the SAME region as "jeans" text.
    "any photo of a dress" lives in the SAME region as "dress" text.

WHY is this powerful? Because you never had to label anything.
The captions on the internet ARE the labels. CLIP learns the whole
visual vocabulary of human fashion — for free.

  After training:
    image of ANY blue thing  →  vector near text "blue"
    image of ANY short skirt →  vector near text "mini skirt"

  Zero new training needed. That's why it's called ZERO-SHOT.
""")
input("  Press Enter to continue...")

if CLIP_AVAILABLE:
    print("  Let's verify this: encode an image and a matching text,")
    print("  then show they land near each other in 512-d space.\n")

    # Synthetic "blue" image
    blue_img = Image.new("RGB", (224, 224), color=(60, 100, 200))
    draw = ImageDraw.Draw(blue_img)
    draw.rectangle([60, 20, 160, 200], fill=(40, 70, 180))

    img_inputs = _clip_processor(images=blue_img, return_tensors="pt")
    with torch.no_grad():
        vision_out = _clip_model.vision_model(pixel_values=img_inputs["pixel_values"])
        img_embed  = _clip_model.visual_projection(vision_out.pooler_output)
        img_embed  = img_embed / img_embed.norm(dim=-1, keepdim=True)

    text_labels = ["blue jeans", "red dress", "white sneakers", "green shorts"]
    txt_inputs  = _clip_processor(text=text_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_out    = _clip_model.text_model(
            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs["attention_mask"],
        )
        txt_embeds = _clip_model.text_projection(txt_out.pooler_output)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

    print("  Similarity of [blue image] to each text label:")
    for i, label in enumerate(text_labels):
        sim = (img_embed @ txt_embeds[i:i+1].T).item()
        bar = "█" * int(sim * 50)
        print(f"    '{label}': {sim:.4f}  {bar}")
    print()
    print("  'blue jeans' wins — the image and text land closest together.")
    print()

input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3: What Is Cosine Similarity? (The Math Behind It)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We've been saying "how close are two vectors." Now let's see the actual math.

PROBLEM: Two 512-d vectors. How do we measure if they point in the same
         "direction" in that high-dimensional space?

ANSWER: Cosine similarity.

  cosine_similarity(A, B) = (A · B) / (|A| × |B|)

  Where:
    A · B = dot product (multiply each pair of elements, then sum)
    |A|   = the "length" (L2 norm) of vector A = sqrt(sum of squares)

  WHAT IT RETURNS:
     1.0  → vectors point in IDENTICAL direction  (same meaning)
     0.0  → vectors are PERPENDICULAR             (unrelated)
    -1.0  → vectors point in OPPOSITE directions  (opposite meaning)

Let's compute it by hand with 3-d vectors (same idea works in 512-d):
""")

# Show the math step by step with actual numbers
A = [0.6, 0.8, 0.0]
B = [0.5, 0.85, 0.1]   # slightly different but in same direction
C = [0.0, 0.1, 0.99]   # completely different

def dot_product(a, b):
    result = sum(x * y for x, y in zip(a, b))
    return result

def l2_norm(v):
    return math.sqrt(sum(x**2 for x in v))

def cosine_sim(a, b):
    return dot_product(a, b) / (l2_norm(a) * l2_norm(b))

print(f"  Vector A = {A}   (imagine: 'blue jeans')")
print(f"  Vector B = {B}   (imagine: 'denim pants')")
print(f"  Vector C = {C}   (imagine: 'pizza')")
print()
print(f"  Step 1 — Dot product of A and B:")
pairs = [f"({A[i]:.1f}×{B[i]:.2f})" for i in range(3)]
print(f"    A·B = {' + '.join(pairs)} = {dot_product(A, B):.4f}")
print()
print(f"  Step 2 — L2 norms:")
print(f"    |A| = sqrt({A[0]}² + {A[1]}² + {A[2]}²) = sqrt({sum(x**2 for x in A):.2f}) = {l2_norm(A):.4f}")
print(f"    |B| = sqrt({B[0]}² + {B[1]}² + {B[2]}²) = sqrt({sum(x**2 for x in B):.2f}) = {l2_norm(B):.4f}")
print()
print(f"  Step 3 — Cosine similarity:")
print(f"    cosine(A,B) = {dot_product(A,B):.4f} / ({l2_norm(A):.4f} × {l2_norm(B):.4f})")
print(f"               = {dot_product(A,B):.4f} / {l2_norm(A)*l2_norm(B):.4f}")
print(f"               = {cosine_sim(A,B):.4f}  (close to 1.0 → similar!)")
print()
print(f"  cosine(A,C) = {cosine_sim(A,C):.4f}  (close to 0.0 → unrelated)")
print()
print("  WHY DO WE L2-NORMALIZE FIRST?")
print()
print("  If we normalize every vector to length 1 (divide by its own norm),")
print("  then |A| = |B| = 1, so the formula simplifies to:")
print()
print("    cosine_similarity(A, B) = A · B  ← just the dot product!")
print()
print("  This is MUCH faster in code, and it's why you always see:")
print("    embed = embed / embed.norm(dim=-1, keepdim=True)")
print("  in ML code. We're making every vector a 'unit vector' (length 1)")
print("  so that dot product = cosine similarity automatically.")

# Show normalization
def normalize(v):
    n = l2_norm(v)
    return [x / n for x in v]

A_norm = normalize(A)
B_norm = normalize(B)
print()
print(f"  A normalized = {[round(x, 4) for x in A_norm]}")
print(f"  B normalized = {[round(x, 4) for x in B_norm]}")
print(f"  norm of A_norm = {l2_norm(A_norm):.4f}  (exactly 1.0 ✓)")
print(f"  A_norm · B_norm = {dot_product(A_norm, B_norm):.4f}  (same as cosine sim above ✓)")
print()
input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4: Zero-Shot Classification — Step by Step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Zero-shot" means: classify without ANY labeled training examples.
You just describe the categories in plain English. CLIP does the rest.

The 4 steps are:

  STEP 1: Encode the image → get a 512-d vector
  STEP 2: Encode each label text → get a 512-d vector per label
  STEP 3: Compute cosine similarity between image and each label
  STEP 4: Run softmax → convert scores to probabilities → pick highest

Let's run all 4 steps with real CLIP right now.
""")
input("  Press Enter to run steps 1-4...")

labels = [
    "blue jeans",
    "red dress",
    "white sneakers",
    "black leather jacket",
    "green shorts",
    "yellow sundress",
]

if CLIP_AVAILABLE:
    # Create synthetic test image: blue rectangle (looks like jeans color)
    img = Image.new("RGB", (224, 224), color=(55, 90, 175))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 20, 165, 205], fill=(35, 65, 155))

    print("\n─── STEP 1: Encode the image ───\n")
    t0 = time.time()
    img_inputs = _clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        vision_out  = _clip_model.vision_model(pixel_values=img_inputs["pixel_values"])
        raw_feature = vision_out.pooler_output         # shape: [1, 768]
        img_embed   = _clip_model.visual_projection(raw_feature)  # shape: [1, 512]
        img_embed   = img_embed / img_embed.norm(dim=-1, keepdim=True)  # L2 normalize
    t1 = time.time()

    print(f"  Input image:  224×224 pixels  →  3 color channels  →  {224*224*3:,} raw numbers")
    print(f"  After ViT encoder:  raw features shape = {list(raw_feature.shape)}")
    print(f"  After projection:   embedding shape   = {list(img_embed.shape)}")
    print(f"  After L2 normalize: norm = {img_embed.norm().item():.4f}  (should be 1.0)")
    print(f"  First 8 values of embedding: {[round(x,4) for x in img_embed[0,:8].tolist()]}")
    print(f"  Time taken: {(t1-t0)*1000:.1f} ms")

    print("\n─── STEP 2: Encode each text label ───\n")
    t0 = time.time()
    txt_inputs = _clip_processor(text=labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_out    = _clip_model.text_model(
            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs["attention_mask"],
        )
        txt_embeds = _clip_model.text_projection(txt_out.pooler_output)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
    t1 = time.time()

    print(f"  {len(labels)} labels encoded in one batch → shape {list(txt_embeds.shape)}")
    print(f"  Each row = one label's 512-d embedding (L2 normalized)")
    print(f"  Time taken: {(t1-t0)*1000:.1f} ms")

    print("\n─── STEP 3: Cosine similarity scores ───\n")
    # img_embed shape: [1, 512], txt_embeds shape: [6, 512]
    # Matrix multiply: [1,512] × [512,6] = [1,6] → one score per label
    raw_scores = (img_embed @ txt_embeds.T)[0]  # shape: [6]
    print("  Cosine similarity (before temperature scaling):")
    for label, score in zip(labels, raw_scores):
        print(f"    '{label}': {score.item():.4f}")

    print()
    print("  We multiply by 100 (temperature scaling) before softmax.")
    print("  WHY? Raw cosine scores are all bunched near 0.2-0.3.")
    print("  Multiplying by 100 spreads them out so softmax gives clear probabilities.")
    logits = 100.0 * raw_scores
    print(f"  After ×100: {[round(x.item(),2) for x in logits]}")

    print("\n─── STEP 4: Softmax → probabilities → winner ───\n")
    print("  WHAT IS SOFTMAX?")
    print("  It converts raw scores into probabilities that add up to 1.0.")
    print("  Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)")
    print()
    print("  Step-by-step with the first 2 values:")
    s0, s1 = logits[0].item(), logits[1].item()
    print(f"    logit[0] ({labels[0]}) = {s0:.2f}  → exp({s0:.2f}) = {math.exp(min(s0,20)):.4f}")
    print(f"    logit[1] ({labels[1]}) = {s1:.2f}  → exp({s1:.2f}) = {math.exp(min(s1,20)):.4f}")
    print()

    probs = logits.softmax(dim=-1)
    print("  Final probabilities:")
    print()
    print(f"  {'Label':<25} | {'Score':>7} | {'Probability':>11} | Bar")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*11}-+--{'-'*20}")
    for label, prob in zip(labels, probs):
        bar = "█" * int(prob.item() * 60)
        print(f"  {label:<25} | {raw_scores[labels.index(label)].item():>7.4f} | {prob.item():>10.2%}  | {bar}")

    winner_idx = probs.argmax().item()
    print()
    print(f"  PREDICTION: '{labels[winner_idx]}' with {probs[winner_idx].item():.1%} confidence")

else:
    print("  (CLIP not available — showing toy numbers to illustrate the concept)\n")
    toy_scores = [0.82, 0.31, 0.19, 0.22, 0.15, 0.20]
    print("  Simulated cosine scores:")
    for label, score in zip(labels, toy_scores):
        print(f"    '{label}': {score:.2f}")
    total = sum(math.exp(s * 100) for s in toy_scores)
    probs = [math.exp(s * 100) / total for s in toy_scores]
    print("\n  Softmax probabilities:")
    for label, prob in zip(labels, probs):
        bar = "█" * int(prob * 50)
        print(f"    '{label}': {prob:.1%}  {bar}")
    print(f"\n  PREDICTION: '{labels[0]}' wins")

print()
input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5: What main.py Does on Every Request
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now that you understand the 4-step process, let's trace what happens
when a user uploads a photo to your app. main.py does NOT call CLIP once.
It calls CLIP MANY TIMES per image.

Here's the flow of predict_top() — the core function:

  def predict_top(image, labels, top_k=1):
      # Package image + labels together for CLIP
      inputs = processor(text=labels, images=image,
                         return_tensors="pt", padding=True)

      # One forward pass through CLIP (steps 1-3 from Chapter 4)
      with torch.no_grad():
          outputs = model(**inputs)
          probs = outputs.logits_per_image.softmax(dim=1)[0]

      # Return top_k highest-scoring labels
      top_indices = probs.topk(top_k).indices.tolist()
      return [(labels[i], probs[i].item()) for i in top_indices]

And detect_outfit() calls predict_top() in multiple rounds:
""")

if CLIP_AVAILABLE:
    img = Image.new("RGB", (224, 224), color=(55, 90, 175))

    print("  Timing each CLIP call on this machine:\n")

    call_descriptions = [
        ("Category detection (12 labels)", ["a t-shirt or casual top","a blouse or dressy top",
         "a sweater or hoodie","a jacket, coat, or blazer","jeans or denim pants",
         "trousers or dress pants","shorts","a dress","a skirt",
         "sneakers or athletic shoes","boots","sandals or heels"]),
        ("Color detection (12 labels)", ["white jeans","black jeans","blue jeans",
         "red jeans","pink jeans","green jeans","beige jeans","brown jeans",
         "gray jeans","yellow jeans","purple jeans","orange jeans"]),
        ("Style: fit (7 labels)", ["wide leg jeans","straight leg jeans","skinny jeans",
         "slim fit jeans","baggy jeans","flared jeans","bootcut jeans"]),
        ("Style: rise (3 labels)", ["high waist jeans","mid rise jeans","low rise jeans"]),
        ("Style: detail (4 labels)", ["distressed ripped jeans","plain solid jeans",
         "embroidered jeans","patchwork jeans"]),
        ("Overall style (8 labels)", ["casual outfit","formal outfit","business casual outfit",
         "streetwear outfit","athleisure outfit","minimal outfit","summer outfit","party outfit"]),
    ]

    total_ms = 0
    for desc, lbs in call_descriptions:
        t0 = time.time()
        inputs = _clip_processor(text=lbs, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            _ = _clip_model(**inputs)
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed
        print(f"  {desc:<40} → {elapsed:6.1f} ms")

    print(f"\n  {'TOTAL per request':<40} → {total_ms:6.1f} ms  (~{total_ms/1000:.2f} seconds)")
    print()
    print("  And that's for ONE clothing item. A full outfit (top + bottom + shoes)")
    print("  multiplies all the style-attribute calls by 3.")
    print()
    print("  A fine-tuned single-pass model would replace ALL of this with one ~20ms call.")
    print("  That's the payoff of Lesson 5.")

else:
    print("  (CLIP not available, but here's what the timing looks like conceptually:)\n")
    print("  Category detection (12 labels)           →  ~80 ms")
    print("  Color detection    (12 labels)           →  ~80 ms")
    print("  Style: fit         (7 labels)            →  ~60 ms")
    print("  Style: rise        (3 labels)            →  ~50 ms")
    print("  Style: detail      (4 labels)            →  ~55 ms")
    print("  Overall style      (8 labels)            →  ~65 ms")
    print("  ─────────────────────────────────────────────────────")
    print("  TOTAL per request                        → ~400 ms")
    print()
    print("  Fine-tuned single-pass model             →  ~20 ms  (20× faster!)")

print()
input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 6: The Ceiling of Zero-Shot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Zero-shot is remarkably powerful but has real limits. Let's SHOW them.

PROBLEM 1: Label wording matters too much.
────────────────────────────────────────
  "wide leg jeans" and "relaxed fit jeans" mean the same thing to a human.
  To CLIP they might score very differently because CLIP learned from web
  captions that used one phrase more than the other.

  Watch what happens with these phrasings of the same concept:
""")

if CLIP_AVAILABLE:
    # Show sensitivity to label wording
    img = Image.new("RGB", (224, 224), color=(55, 90, 175))
    draw = ImageDraw.Draw(img)
    draw.rectangle([55, 15, 170, 210], fill=(35, 65, 155))

    phrase_groups = [
        ("Wide-leg jeans described different ways:", [
            "wide leg jeans",
            "relaxed fit jeans",
            "baggy jeans",
            "loose fit denim",
            "wide cut denim pants",
        ]),
        ("Skinny jeans described different ways:", [
            "skinny jeans",
            "slim fit jeans",
            "tight jeans",
            "fitted denim",
            "tapered jeans",
        ]),
    ]

    for title, phrase_list in phrase_groups:
        print(f"\n  {title}")
        inputs = _clip_processor(text=phrase_list, images=img,
                                  return_tensors="pt", padding=True)
        with torch.no_grad():
            out   = _clip_model(**inputs)
            probs = out.logits_per_image.softmax(dim=1)[0]
        for phrase, prob in zip(phrase_list, probs):
            bar = "█" * int(prob.item() * 80)
            print(f"    {phrase:<30} {prob.item():.1%}  {bar}")

    print("""
  Notice how much the score changes based on exact wording.
  This is called "prompt sensitivity" — a known weakness of zero-shot models.
  Your app's accuracy depends partly on whoever wrote the label strings in main.py.
""")

else:
    print("  (CLIP unavailable — simulating prompt sensitivity)\n")
    print("  'wide leg jeans'     → 45%")
    print("  'relaxed fit jeans'  → 28%")
    print("  'baggy jeans'        → 15%")
    print("  'loose fit denim'    → 8%")
    print("  'wide cut denim'     → 4%")
    print()
    print("  Different words for the SAME thing get wildly different scores.")

print("""
PROBLEM 2: Ambiguous images confuse the whole-image embedding.
────────────────────────────────────────────────────────────
  CLIP embeds the ENTIRE image into ONE vector.
  If someone wears blue jeans + red top, that one vector must encode BOTH.
  When you ask "what color is this?" — CLIP averages over the whole image.

  main.py's clever workaround: it asks "white jeans? blue jeans? ..."
  This forces CLIP to think about the jeans specifically (because "jeans"
  is in the label), which partially disambiguates. But it's still imperfect.

PROBLEM 3: CLIP doesn't know fashion taxonomy.
──────────────────────────────────────────────
  CLIP was trained on ALL of the internet, not just fashion sites.
  It doesn't know that "midi" specifically means skirt/dress length.
  It doesn't know your app's exact category definitions.
  A fine-tuned model trained on YOUR dataset would.

PROBLEM 4: Speed (you just measured this in Chapter 5).
""")
input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 7: What Fine-Tuning Will Give Us (Preview)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fine-tuning means: take CLIP's learned features and teach it YOUR
specific categories using labeled training examples.

Here's the architecture preview:

  Input image (224×224)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  CLIP Vision Encoder  (86M parameters)      │
  │  Frozen in phase 1 → slowly unfrozen in 2  │
  │  Already knows: colors, textures, shapes   │
  └──────────────────────────┬──────────────────┘
                             │ 512-d embedding
                             ▼
                   ┌──────── L2 normalize ────────┐
                   │                              │
                   ▼                              ▼
         ┌─────────────────┐           ┌─────────────────┐
         │  Category Head  │           │   Color Head    │
         │  Linear(512,256)│           │  Linear(512,256)│
         │  ReLU           │           │  ReLU           │
         │  Dropout(0.2)   │           │  Dropout(0.2)   │
         │  Linear(256,N)  │           │  Linear(256,M)  │
         └────────┬────────┘           └────────┬────────┘
                  │                             │
                  ▼                             ▼
           "jeans" / "dress"            "blue" / "red" / ...
              (one forward pass!)           (same pass!)

ADVANTAGES over zero-shot:

  1. ONE forward pass → ALL predictions simultaneously
     Instead of 6+ CLIP calls: one pass gives category + color together.

  2. Fashion-specific features
     CLIP backbone adapts to notice fashion-specific patterns
     (e.g., denim texture, neckline shapes, waistband details).

  3. Consistent vocabulary
     The model learns YOUR category names directly from YOUR dataset.
     "midi skirt" always means what YOUR data says it means.

  4. Speed
     ~20ms instead of ~400ms per image.

PARAMETER COUNTS (reality check):
""")

if CLIP_AVAILABLE:
    backbone_params = sum(p.numel() for p in _clip_model.vision_model.parameters())
    proj_params     = sum(p.numel() for p in _clip_model.visual_projection.parameters())
    print(f"  CLIP vision encoder:          {backbone_params:>12,} parameters")
    print(f"  CLIP visual projection:       {proj_params:>12,} parameters")
    # Estimate head sizes
    import torch.nn as nn
    head_cat = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,10))
    head_col = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,13))
    head_params = sum(p.numel() for p in head_cat.parameters()) + \
                  sum(p.numel() for p in head_col.parameters())
    print(f"  Our classification heads:     {head_params:>12,} parameters")
    print(f"  Total:                        {backbone_params+proj_params+head_params:>12,} parameters")
    print()
    pct = head_params / (backbone_params + proj_params + head_params) * 100
    print(f"  Our new code is {pct:.1f}% of the total model.")
    print(f"  We're adding a tiny translation layer onto a massive pretrained foundation.")
else:
    print("  CLIP vision encoder:           ~86,000,000 parameters")
    print("  CLIP visual projection:            262,656 parameters")
    print("  Our classification heads:           ~50,000 parameters")
    print()
    print("  Our new code is ~0.06% of the total model.")

print("""
  This is the key insight of TRANSFER LEARNING:
  You don't train from scratch. You borrow 86M parameters of learned knowledge
  and add a small ~50K parameter "translation layer" on top.

  Run python learn/05_fine_tune.py to build and train this model.
  (Requires the Myntra dataset — instructions in that file.)
""")
input("  Press Enter to finish...")

print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Lesson 4 complete.")
print("  What you learned:")
print("    ✓ Embeddings = lists of numbers where similar things land near each other")
print("    ✓ CLIP = trained on 400M image-text pairs via contrastive learning")
print("    ✓ Cosine similarity = dot product of unit vectors (angle between them)")
print("    ✓ Zero-shot classification = encode image → encode labels → pick closest")
print("    ✓ main.py calls CLIP 6+ times per image (slow)")
print("    ✓ Fine-tuning adds small heads → 1 pass instead of 6+")
print()
print("  Next: python learn/05_fine_tune.py")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
