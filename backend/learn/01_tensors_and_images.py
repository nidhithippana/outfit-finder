"""
LESSON 1: What is a tensor? What does an image actually look like to a computer?

No assumptions. Start from scratch.

Run: python learn/01_tensors_and_images.py
"""

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 1: Numbers are ALL a computer knows
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1: The only thing a computer understands is numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An AI model NEVER sees an image the way you do.
It never sees "jeans" or "blue" or "a jacket."

It only ever sees numbers. Thousands of them. In a grid.

Your job as an ML engineer is to turn real-world things
(photos, words, prices) into grids of numbers — and then
train a model to find patterns in those numbers.

A TENSOR is just a word for "a grid of numbers."
That's it. Nothing scary.

  - A single number               = 0D tensor  (called a "scalar")
  - A list of numbers             = 1D tensor  (called a "vector")
  - A table (rows and columns)    = 2D tensor  (called a "matrix")
  - A cube (rows, cols, depth)    = 3D tensor
  - A stack of cubes              = 4D tensor  ← what a batch of images is
""")

input("  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2: Building tensors by hand
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2: Let's build tensors by hand
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ── Scalar ────────────────────────────────────────────────────────────────────
print("── Scalar (0D) ─────────────────────────────────────────────")
print("Just one number. Like a price, a confidence score, a loss value.\n")

temperature = torch.tensor(98.6)
print(f"  temperature = {temperature}")
print(f"  shape = {temperature.shape}   ← empty means 0 dimensions (scalar)")
print(f"  value = {temperature.item()}")   # .item() extracts the raw Python number

input("\n  Press Enter to continue...")

# ── Vector ────────────────────────────────────────────────────────────────────
print("\n── Vector (1D) ─────────────────────────────────────────────")
print("A list of numbers. Like RGB color values, or pixel brightness along one row.\n")

red_pixel = torch.tensor([255.0, 0.0, 0.0])
print(f"  A pure red pixel in RGB: {red_pixel}")
print(f"  Shape: {red_pixel.shape}   ← 3 numbers, 1 dimension")
print()

blue_pixel = torch.tensor([0.0, 0.0, 255.0])
green_pixel = torch.tensor([0.0, 255.0, 0.0])
print(f"  Blue pixel:  {blue_pixel}")
print(f"  Green pixel: {green_pixel}")
print()
print("  RGB = Red, Green, Blue. Every color on your screen is 3 numbers.")
print("  Pure red   = [255, 0,   0  ]")
print("  Pure green = [0,   255, 0  ]")
print("  Pure blue  = [0,   0,   255]")
print("  White      = [255, 255, 255]")
print("  Black      = [0,   0,   0  ]")
print("  Yellow     = [255, 255, 0  ]  (red + green mixed)")

input("\n  Press Enter to continue...")

# ── Matrix ────────────────────────────────────────────────────────────────────
print("\n── Matrix (2D) ─────────────────────────────────────────────")
print("A table of numbers. Like a grayscale image (just brightness, no color).\n")
print("Imagine a tiny 4x4 black-and-white photo:")
print("  0 = black,  255 = white,  128 = gray\n")

grayscale_image = torch.tensor([
    [  0,   0,   0,   0],   # row 0: all black
    [  0, 255, 255,   0],   # row 1: white in the middle
    [  0, 255, 255,   0],   # row 2: white in the middle
    [  0,   0,   0,   0],   # row 3: all black
], dtype=torch.float32)

print(f"  Grayscale 4x4 image:\n{grayscale_image}")
print(f"\n  Shape: {grayscale_image.shape}  ← 4 rows, 4 columns")
print("\n  What you'd see: a tiny white square on a black background.")

input("\n  Press Enter to continue...")

# ── 3D Tensor ─────────────────────────────────────────────────────────────────
print("\n── 3D Tensor ───────────────────────────────────────────────")
print("A COLOR image. 3 layers (channels) stacked on top of each other.\n")
print("""
  Think of it like a club sandwich:

  ┌──────────────┐
  │  RED layer   │  ← how much red in each pixel
  ├──────────────┤
  │ GREEN layer  │  ← how much green in each pixel
  ├──────────────┤
  │  BLUE layer  │  ← how much blue in each pixel
  └──────────────┘

  Each layer is a 2D grid (height × width).
  Stack 3 of them → shape is [3, height, width]

  For a 224×224 photo: shape = [3, 224, 224]
  That's 3 × 224 × 224 = 150,528 numbers for ONE image.
""")

# A 2x2 color image where:
# top-left = red, top-right = green, bottom-left = blue, bottom-right = white
color_image = torch.zeros(3, 2, 2)  # 3 channels, 2x2

# Set red channel (channel 0)
color_image[0, 0, 0] = 1.0   # top-left red: max red
color_image[0, 0, 1] = 0.0   # top-right green: no red
color_image[0, 1, 0] = 0.0   # bottom-left blue: no red
color_image[0, 1, 1] = 1.0   # bottom-right white: max red

# Set green channel (channel 1)
color_image[1, 0, 0] = 0.0
color_image[1, 0, 1] = 1.0   # top-right green: max green
color_image[1, 1, 0] = 0.0
color_image[1, 1, 1] = 1.0   # bottom-right white: max green

# Set blue channel (channel 2)
color_image[2, 0, 0] = 0.0
color_image[2, 0, 1] = 0.0
color_image[2, 1, 0] = 1.0   # bottom-left blue: max blue
color_image[2, 1, 1] = 1.0   # bottom-right white: max blue

print("  A 2x2 image (red | green / blue | white):\n")
print("  Red channel (how much red per pixel):")
print(f"    {color_image[0]}")
print("  Green channel:")
print(f"    {color_image[1]}")
print("  Blue channel:")
print(f"    {color_image[2]}")
print(f"\n  Full tensor shape: {color_image.shape}  = [channels, height, width]")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3: A real image — from file to tensor, step by step
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3: Loading a real image — every step explained
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Make a test image: blue rectangle (like jeans) on white background
img = Image.new("RGB", (100, 150), color=(240, 240, 240))   # white background
draw = ImageDraw.Draw(img)
draw.rectangle([20, 10, 80, 140], fill=(60, 100, 200))      # blue jeans shape
img.save("/tmp/test_outfit.jpg")

print("  Created a test image: white background with a blue rectangle.")
print("  Saved to /tmp/test_outfit.jpg\n")

# ── Step 1: PIL opens the file ────────────────────────────────────────────────
print("  STEP 1: PIL (Python Imaging Library) opens the file")
print("  ─────────────────────────────────────────────────────")
pil_img = Image.open("/tmp/test_outfit.jpg").convert("RGB")
print(f"  pil_img.mode  = '{pil_img.mode}'  (RGB = 3 color channels)")
print(f"  pil_img.size  = {pil_img.size}    (width × height)")
print(f"  Type: {type(pil_img)}")
print()
print("  PIL stores images as (width, height).")
print("  PyTorch wants (channels, height, width).")
print("  So we need to reorder things. That's what the transform does.\n")

input("  Press Enter to continue...")

# ── Step 2: What are the actual pixel values? ─────────────────────────────────
print("\n  STEP 2: What are the actual pixel values?")
print("  ─────────────────────────────────────────────────────")
pixel_array = np.array(pil_img)
print(f"  numpy array shape: {pixel_array.shape}  ← (height, width, channels)")
print(f"  data type:         {pixel_array.dtype}  ← 8-bit unsigned integers")
print(f"  value range:       {pixel_array.min()} to {pixel_array.max()}")
print()
print("  Corner pixels (row, col):")
print(f"    top-left     [0,0]  = {pixel_array[0, 0]}   ← white background")
print(f"    center       [75,50]= {pixel_array[75, 50]}  ← blue jeans area")
print()
print("  Each pixel = 3 integers from 0–255.")
print("  [240, 240, 240] = almost-white (high R, G, B equally)")
print("  [ 60, 100, 200] = blue-ish (high B, medium G, low R)")

input("\n  Press Enter to continue...")

# ── Step 3: ToTensor ──────────────────────────────────────────────────────────
print("\n  STEP 3: ToTensor() — integers to floats, reshape")
print("  ─────────────────────────────────────────────────────")
print()
print("  ToTensor() does TWO things automatically:")
print("  1. Divides every pixel by 255  → values go from 0–255 to 0.0–1.0")
print("  2. Reorders axes from (H,W,C) to (C,H,W)  ← PyTorch convention")
print()

to_tensor = T.ToTensor()
tensor = to_tensor(pil_img)

print(f"  Before: numpy shape = {pixel_array.shape}  (height, width, channels)")
print(f"  After:  tensor shape = {tensor.shape}   (channels, height, width)")
print()
print(f"  Before: pixel value = {pixel_array[75, 50]}  (integers 0–255)")
# After ToTensor, pixel at same location:
print(f"  After:  pixel value = [{tensor[0,75,50]:.3f}, {tensor[1,75,50]:.3f}, {tensor[2,75,50]:.3f}]  (floats 0.0–1.0)")
print()
print("  WHY 0.0–1.0?  Models work best when inputs are small numbers.")
print("  255 as input causes unstable gradients. 0.75 is fine.")

input("\n  Press Enter to continue...")

# ── Step 4: Resize ────────────────────────────────────────────────────────────
print("\n  STEP 4: Resize — all images must be the same size")
print("  ─────────────────────────────────────────────────────")
print()
print("  Your photos might be 100×150, 1920×1080, or 500×500.")
print("  The model expects exactly 224×224. CLIP was trained on 224×224.")
print("  So we resize everything to 224×224 before feeding the model.")
print()

resize = T.Resize(224)
resized = resize(pil_img)
print(f"  Original size: {pil_img.size}  (width × height)")
print(f"  After resize:  {resized.size}")
print()
print("  Resize(224) scales the SHORTEST edge to 224.")
print("  Then CenterCrop(224) cuts out a 224×224 square from the center.")

input("\n  Press Enter to continue...")

# ── Step 5: Normalize ─────────────────────────────────────────────────────────
print("\n  STEP 5: Normalize — shift values so the model is stable")
print("  ─────────────────────────────────────────────────────")
print()
print("  After ToTensor, values are 0.0–1.0.")
print("  After Normalize, values are roughly -2.5 to +2.5.")
print()
print("  The formula: (pixel - mean) / std")
print("  Applied separately to each color channel.")
print()
print("  CLIP was trained with these specific values:")
print("    mean = [0.4815, 0.4578, 0.4082]  ← average color of the web")
print("    std  = [0.2686, 0.2613, 0.2758]  ← how spread out the values are")
print()
print("  WHY does this matter?")
print("  Neural network weights are initialized near 0.")
print("  If inputs are all positive (0–1), gradients all push the same direction.")
print("  Centering around 0 (roughly -2.5 to +2.5) makes training symmetric and stable.")
print()

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275,  0.40821073],
    std= [0.26862954, 0.26130258, 0.27577711],
)
tensor_float = T.ToTensor()(T.Resize(224)(T.CenterCrop(224)(T.Resize(224)(pil_img))))
normalized = normalize(tensor_float)

print(f"  Before normalize: min={tensor_float.min():.2f}, max={tensor_float.max():.2f}")
print(f"  After normalize:  min={normalized.min():.2f}, max={normalized.max():.2f}")
print()

# Show one specific pixel's journey
r_before = tensor_float[0, 112, 56].item()
r_after  = normalized[0, 112, 56].item()
print(f"  One pixel's red channel:")
print(f"    Raw file:      {int(r_before * 255)}")
print(f"    After ToTensor: {r_before:.4f}")
print(f"    After Normalize: {r_after:.4f}")
print(f"    Math check: ({r_before:.4f} - 0.4815) / 0.2686 = {(r_before - 0.48145466)/0.26862954:.4f} ✓")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4: Putting the full pipeline together
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4: The full pipeline — what your app runs on every photo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("""
  Every image you upload goes through this exact sequence:

  📸 your photo (any size, any format)
       ↓
  PIL.Image.open()          → Python image object
       ↓
  .convert("RGB")           → make sure it's 3-channel (not grayscale or RGBA)
       ↓
  Resize(224)               → scale shortest edge to 224px
       ↓
  CenterCrop(224)           → cut exact 224×224 square
       ↓
  ToTensor()                → (H,W,C) uint8 [0-255]  →  (C,H,W) float [0.0-1.0]
       ↓
  Normalize(mean, std)      → float [0.0-1.0]  →  float [-2.5 to +2.5]
       ↓
  .unsqueeze(0)             → add batch dimension: [3,224,224] → [1,3,224,224]
       ↓
  model(tensor)             → predictions!
""")

# Run the whole thing
full_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])

result = full_transform(pil_img)           # shape: [3, 224, 224]
batch  = result.unsqueeze(0)               # shape: [1, 3, 224, 224]

print(f"  Input image size:        {pil_img.size}")
print(f"  After full transform:    {result.shape}")
print(f"  After .unsqueeze(0):     {batch.shape}    ← ready for the model")
print()
print(f"  Total numbers in one image: 3 × 224 × 224 = {3*224*224:,}")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5: Batches — why we process many images at once
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5: Batches — why models process images in groups
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("""
  At inference (running your app): you process ONE image at a time.
  At training: you process MANY images at once. This is a "batch."

  WHY batch?
  1. GPUs are built for parallel math. Sending 32 images at once
     is almost as fast as sending 1.
  2. Loss is averaged over the batch, which gives a more stable
     gradient signal than updating after every single image.

  A batch adds a 4th dimension to your tensor:
    One image:       [3, 224, 224]    ← channels, height, width
    Batch of 32:    [32, 3, 224, 224] ← batch, channels, height, width
""")

single = result                                    # [3, 224, 224]
batch4 = torch.stack([result] * 4)                 # pretend 4 images

print(f"  Single image shape:  {single.shape}")
print(f"  Batch of 4 images:   {batch4.shape}")
print()
print("  In train.py, DataLoader builds these batches for you automatically.")
print("  batch_size=32 means 32 images per batch.")
print()
print(f"  Memory for one 224×224 image:  {3*224*224*4 / 1024:.1f} KB")
print(f"  Memory for batch of 32:        {32*3*224*224*4 / 1024 / 1024:.1f} MB")

input("\n  Press Enter to see the summary...")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY — What you just learned
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TENSOR  = a grid of numbers (any number of dimensions)

  scalar  = one number          (0D)
  vector  = a list              (1D)   e.g. [255, 0, 0]
  matrix  = a table             (2D)   e.g. a grayscale image
  3D      = stacked tables      (3D)   e.g. one RGB image [3, H, W]
  4D      = stack of 3D         (4D)   e.g. a batch [B, 3, H, W]

IMAGE PIPELINE:
  .jpg file
    → PIL open          (Python image object)
    → Resize + Crop     (standardize to 224×224)
    → ToTensor()        (H,W,C uint8 → C,H,W float 0–1)
    → Normalize()       (shift to ~-2.5 to +2.5)
    → .unsqueeze(0)     (add batch dimension)
    → ready for model ✓

NEXT: Lesson 2 teaches what the model does with these numbers.
""")

print("✓ Lesson 1 complete. Run: python learn/02_training_loop.py")
