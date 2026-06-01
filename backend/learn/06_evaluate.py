"""
LESSON 6: Evaluation — knowing if your model is actually good.

No assumptions. Start from absolute zero.

This lesson answers the most important question in machine learning:
"My model trained. But is it actually good?"

Run:
  python learn/06_evaluate.py

Works even if you haven't trained a model yet — we use synthetic predictions
to demonstrate every concept with real math.
"""

import sys
import random
import math
from pathlib import Path
from collections import defaultdict, Counter

import torch

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 1: Why "accuracy" alone can lie to you
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1: Why "accuracy" alone can lie to you
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Let's say you're building a fashion classifier.
Your training dataset looks like this:

  Jeans:   5000 images   (collected easily from large retailers)
  Skirts:    50 images   (harder to find, so fewer were collected)

Total: 5050 images.

Now imagine the laziest possible "model":
  It looks at EVERY image and always predicts "jeans."
  It never tries. It just always says "jeans."

How does that model score?
""")

total_images = 5050
jeans_images = 5000
skirt_images = 50

lazy_correct = jeans_images   # got all jeans right by luck
lazy_wrong   = skirt_images   # got all skirts wrong
lazy_accuracy = lazy_correct / total_images

print(f"  Total images:        {total_images}")
print(f"  Always-jeans correct: {lazy_correct}  (all jeans images)")
print(f"  Always-jeans wrong:   {lazy_wrong}   (all skirt images)")
print()
print(f"  Accuracy = correct / total")
print(f"  Accuracy = {lazy_correct} / {total_images}")
print(f"  Accuracy = {lazy_accuracy:.4f}")
print(f"  Accuracy = {lazy_accuracy * 100:.1f}%")

print("""
  That's 98.9% accuracy — and the model is completely useless.
  It cannot identify a single skirt. Zero.

  This is called CLASS IMBALANCE.
  It happens in EVERY real-world fashion dataset because:
    - Jeans and tops are everywhere online
    - Some items (capes, culottes, kaftans) are rare

  If you only report one number — overall accuracy — you can fool yourself
  into thinking you have a great model when you have a broken one.

  The fix: ALWAYS look at per-class accuracy.
  That means one accuracy number for jeans, one for skirts, one for each category.

  Let's see what per-class accuracy looks like for that lazy model:
""")

print("  Per-class accuracy for the 'always predicts jeans' model:")
print()
print(f"    Jeans:   {jeans_images}/{jeans_images} correct = 100.0%  ← looks amazing")
print(f"    Skirts:       0/{skirt_images}  correct =   0.0%  ← completely broken")
print()
print("  That 0% on skirts is the truth the 98.9% was hiding.")
print()
print("  Rule: Never trust a single accuracy number. Always break it down by class.")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2: Precision, Recall, and F1 — the three real metrics
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2: Precision, Recall, F1 — the three real metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per-class accuracy is better than overall accuracy.
But even that isn't the full picture. We need three more numbers.

First, meet the 2x2 confusion box. Pick one class — say "jacket".
Every single prediction your model makes is one of four things:

                        WHAT THE MODEL PREDICTED
                      ┌──────────────┬──────────────┐
                      │  "jacket"    │  not "jacket" │
          ┌───────────┼──────────────┼──────────────┤
  WHAT    │  jacket   │   TRUE       │   FALSE       │
  THE     │  (actual) │  POSITIVE    │  NEGATIVE     │
  IMAGE   │           │  (TP)        │  (FN)         │
  REALLY  ├───────────┼──────────────┼──────────────┤
  IS      │ not jacket│   FALSE      │   TRUE        │
          │  (actual) │  POSITIVE    │  NEGATIVE     │
          │           │  (FP)        │  (TN)         │
          └───────────┴──────────────┴──────────────┘

  TRUE POSITIVE  (TP): Model said "jacket", image IS a jacket.     ✓ WIN
  FALSE POSITIVE (FP): Model said "jacket", image is NOT a jacket. ✗ Wrong alert
  TRUE NEGATIVE  (TN): Model said "not jacket", image is NOT.      ✓ WIN
  FALSE NEGATIVE (FN): Model said "not jacket", image IS a jacket. ✗ Missed it

Fashion examples:
  "Predicted jacket, was actually a top"  → False Positive  (said jacket, wasn't)
  "Predicted top, was actually a jacket"  → False Negative  (missed a jacket)
  "Predicted jacket, was a jacket"        → True Positive   (nailed it)

Now let's make up concrete numbers and compute everything by hand.
""")

# Concrete numbers for jacket class
TP = 45   # predicted jacket, was jacket
FP = 8    # predicted jacket, was something else (a top, a dress)
FN = 5    # predicted something else, was actually a jacket
TN = 242  # predicted not-jacket, was not jacket

print("  Scenario: evaluating the 'jacket' class")
print(f"    True Positives  (TP): {TP}  — said jacket, image IS jacket")
print(f"    False Positives (FP): {FP}  — said jacket, image is NOT jacket")
print(f"    False Negatives (FN): {FN}  — said not-jacket, image IS jacket")
print(f"    True Negatives  (TN): {TN} — said not-jacket, image is not jacket")

print("""
  ─────────────────────────────────────────────────────────────
  METRIC 1: PRECISION — "When you predicted jacket, how often were you right?"

  Formula:  Precision = TP / (TP + FP)

  In plain English: of all the times you raised your hand and said "jacket!",
  what fraction of those were actually jackets?

  Like a doctor who only diagnoses a disease when they're really sure.
  High precision = few false alarms.
""")

precision = TP / (TP + FP)
print(f"  Precision = {TP} / ({TP} + {FP})")
print(f"  Precision = {TP} / {TP + FP}")
print(f"  Precision = {precision:.4f}")
print(f"  Precision = {precision * 100:.1f}%")
print()
print(f"  Meaning: when the model says 'jacket', it's right {precision * 100:.1f}% of the time.")

print("""
  ─────────────────────────────────────────────────────────────
  METRIC 2: RECALL — "Of all actual jackets, how many did you catch?"

  Formula:  Recall = TP / (TP + FN)

  In plain English: of all the jacket images in the dataset,
  what fraction did the model actually identify?

  Like a security scanner — you want it to catch every threat,
  even if it occasionally flags something innocent.
  High recall = few missed cases.
""")

recall = TP / (TP + FN)
print(f"  Recall = {TP} / ({TP} + {FN})")
print(f"  Recall = {TP} / {TP + FN}")
print(f"  Recall = {recall:.4f}")
print(f"  Recall = {recall * 100:.1f}%")
print()
print(f"  Meaning: the model finds {recall * 100:.1f}% of all jacket images in the dataset.")

print("""
  ─────────────────────────────────────────────────────────────
  METRIC 3: F1 SCORE — the balance between precision and recall

  Precision and recall trade off against each other:
    - If the model is very conservative (rarely says "jacket"), it has
      HIGH precision but LOW recall (misses many jackets).
    - If the model is trigger-happy (always says "jacket"), it has
      LOW precision but HIGH recall (catches all jackets but cries wolf a lot).

  F1 combines both into one number using the HARMONIC MEAN.
  (Why harmonic mean? Because it punishes extreme imbalances harder than average.)

  Formula:  F1 = 2 * (Precision * Recall) / (Precision + Recall)

  Think of it as: "What's the score if both precision AND recall matter equally?"
  A high F1 means you're both accurate and thorough.
""")

f1 = 2 * (precision * recall) / (precision + recall)
print(f"  F1 = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
print(f"  F1 = 2 * {precision * recall:.4f} / {precision + recall:.4f}")
print(f"  F1 = {2 * precision * recall:.4f} / {precision + recall:.4f}")
print(f"  F1 = {f1:.4f}")
print(f"  F1 = {f1 * 100:.1f}%")

print("""
  ─────────────────────────────────────────────────────────────
  SUMMARY TABLE for the 'jacket' class:
""")

print(f"  ┌─────────────┬─────────┬─────────────────────────────────────────────┐")
print(f"  │ Metric      │  Score  │ Meaning                                     │")
print(f"  ├─────────────┼─────────┼─────────────────────────────────────────────┤")
print(f"  │ Precision   │  {precision*100:5.1f}%  │ When model says jacket, {precision*100:.1f}% right      │")
print(f"  │ Recall      │  {recall*100:5.1f}%  │ Model catches {recall*100:.1f}% of all real jackets  │")
print(f"  │ F1 Score    │  {f1*100:5.1f}%  │ Combined single-number grade                │")
print(f"  └─────────────┴─────────┴─────────────────────────────────────────────┘")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3: The confusion matrix
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3: The confusion matrix — seeing ALL your mistakes at once
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The confusion matrix is the full picture.
Instead of looking at one class at a time, it shows you every single
(predicted class, actual class) combination in one grid.

The rows = what the image actually is.
The columns = what your model predicted.

If every prediction is perfect, only the DIAGONAL has numbers.
Off-diagonal numbers are your mistakes.

Let's build one from scratch with synthetic data for 5 clothing categories.
""")

# Synthetic predictions for 5 classes
CLASSES = ["jacket", "jeans", "dress", "top", "skirt"]
N_CLASSES = len(CLASSES)

# Confusion matrix: rows=actual, cols=predicted
# We're making these numbers up to tell a story
conf_matrix = [
    # jacket  jeans  dress   top  skirt   ← PREDICTED
    [   45,     2,     0,    3,    0  ],  # jacket  ← ACTUAL
    [    1,    89,     0,    2,    0  ],  # jeans
    [    0,     0,    62,   11,    4  ],  # dress
    [    2,     3,     9,   78,    0  ],  # top
    [    0,     0,     7,    4,   31  ],  # skirt
]

print("  Building the confusion matrix...")
print()
print("  Each row = what the image ACTUALLY is")
print("  Each column = what the model PREDICTED")
print()

# Print the matrix nicely
header = "             " + "".join(f"  {c[:6]:>6}" for c in CLASSES)
print(f"  {header}")
print(f"  {'':12}" + "  ------" * N_CLASSES)
for i, row_class in enumerate(CLASSES):
    row_str = f"  {row_class:<12}"
    for j, val in enumerate(conf_matrix[i]):
        if i == j:
            row_str += f"  [{val:>4}]"   # highlight diagonal
        else:
            row_str += f"   {val:>4} "
    print(row_str)

print()
print("  [ ] = diagonal = CORRECT predictions")
print("   x  = off-diagonal = MISTAKES")

print("""
  ─────────────────────────────────────────────────────────────
  How to read this:

  Row "dress" → the model saw 62+11+4 = 77 actual dress images.
    It correctly predicted "dress" for 62 of them.
    But it predicted "top" for 11 dress images.
    And it predicted "skirt" for 4 dress images.

  Row "top" → the model saw 2+3+9+78 = 92 actual top images.
    It correctly predicted "top" for 78 of them.
    But it confused 9 tops as "dress".

  The BIGGEST off-diagonal numbers tell you where your model struggles.
  Let's find them:
""")

biggest_mistake_val = 0
biggest_mistake_info = None
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j and conf_matrix[i][j] > biggest_mistake_val:
            biggest_mistake_val = conf_matrix[i][j]
            biggest_mistake_info = (i, j)

ai, aj = biggest_mistake_info
print(f"  Biggest mistake: {biggest_mistake_val} times the model predicted '{CLASSES[aj]}'")
print(f"                   when the image was actually '{CLASSES[ai]}'")
print()
print(f"  This makes sense! {CLASSES[ai].capitalize()} and {CLASSES[aj]} can look similar:")
print(f"  - Both can be loose-fitting")
print(f"  - Both can have similar lengths in product photos")
print(f"  - Lighting and model pose can blur the difference")

print("""
  ─────────────────────────────────────────────────────────────
  What to DO with this information:

  1. Find the most confused class pair (dress/top here).
  2. Look at the actual misclassified images — open them manually.
     Ask: are these actually mislabeled? Are they genuinely ambiguous?
  3. If mislabeled: fix the labels, retrain.
  4. If genuinely similar: collect more examples that show clear differences.
  5. If too similar to distinguish: consider merging the classes.

  The confusion matrix is your debugging tool. Without it, you're flying blind.
""")

print("  Per-class accuracy from the confusion matrix:")
print()
for i, cls in enumerate(CLASSES):
    row_total = sum(conf_matrix[i])
    correct = conf_matrix[i][i]
    acc = correct / row_total if row_total > 0 else 0
    bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
    print(f"    {cls:<8}  {correct:>3}/{row_total:<3}  {acc*100:>5.1f}%  {bar}")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4: Overfitting vs underfitting — diagnosing from loss curves
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4: Overfitting vs underfitting — reading loss curves
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When you train a model, you watch two loss numbers every epoch:
  - TRAIN LOSS: how wrong the model is on images it's already seen
  - VAL LOSS:   how wrong the model is on images it has NEVER seen

The val loss is the truth. Train loss can be gamed.

Imagine studying for an exam:
  - Train loss = how well you do on practice problems you've already solved
  - Val loss   = how well you do on the real exam with new questions

There are three things that can happen. Let's draw them.
""")

# ASCII loss curve art
def print_loss_curve(title, epochs, train_losses, val_losses):
    print(f"  {title}")
    height = 10
    width  = len(epochs)

    # Normalize to 0..height-1
    all_vals  = train_losses + val_losses
    lo, hi    = min(all_vals), max(all_vals)
    rng       = hi - lo if hi != lo else 1

    def normalize(v):
        return height - 1 - int((v - lo) / rng * (height - 1))

    grid = [[" "] * (width * 2) for _ in range(height)]

    for x, (t, v) in enumerate(zip(train_losses, val_losses)):
        yt = normalize(t)
        yv = normalize(v)
        if 0 <= yt < height:
            grid[yt][x * 2] = "T"
        if 0 <= yv < height:
            grid[yv][x * 2] = "V"

    print(f"  Loss")
    print(f"  {hi:.2f} │")
    for row_i, row in enumerate(grid):
        label = f"       │" if row_i != height - 1 else f"  {lo:.2f} │"
        print(f"  {'       │' if row_i != height - 1 else f'{lo:.2f} │'} {''.join(row)}")
    print(f"         └" + "─" * (width * 2))
    print(f"          Epoch 1" + " " * (width * 2 - 10) + f"Epoch {width}")
    print(f"   T = train loss    V = val loss")
    print()


print("  ─────────────────────────────────────────────────────────────")
print("  SITUATION 1: UNDERFITTING")
print("  Both train AND val loss are stuck high. The model isn't learning.")
print()
underfit_train = [1.8, 1.75, 1.72, 1.70, 1.69, 1.68, 1.67, 1.66]
underfit_val   = [1.82, 1.77, 1.74, 1.72, 1.71, 1.70, 1.69, 1.68]
epochs = list(range(1, len(underfit_train) + 1))
print_loss_curve("  Underfitting:", epochs, underfit_train, underfit_val)

print("  What this means:")
print("    - The model is too simple for the problem")
print("    - OR: you haven't trained long enough")
print("    - OR: your learning rate is too low (barely updating)")
print()
print("  What to do:")
print("    - Train for more epochs")
print("    - Increase the learning rate slightly")
print("    - Unfreeze more layers of the backbone")
print("    - Use a larger model (vit-large instead of vit-base)")
print()

print("  ─────────────────────────────────────────────────────────────")
print("  SITUATION 2: THE SWEET SPOT (what you want)")
print("  Both losses drop. Val loss stays close to train loss. Both level off.")
print()
good_train = [1.8, 1.2, 0.85, 0.62, 0.50, 0.43, 0.40, 0.38]
good_val   = [1.85, 1.28, 0.90, 0.68, 0.56, 0.50, 0.48, 0.47]
print_loss_curve("  Good fit:", epochs, good_train, good_val)

print("  What this means:")
print("    - The model is learning real patterns, not memorizing")
print("    - The gap between train and val is small")
print("    - This is what you want to see")
print()

print("  ─────────────────────────────────────────────────────────────")
print("  SITUATION 3: OVERFITTING (the most common problem)")
print("  Train loss keeps dropping. Val loss starts going BACK UP.")
print()
over_train = [1.8, 1.2, 0.85, 0.60, 0.42, 0.28, 0.17, 0.09]
over_val   = [1.85, 1.28, 0.90, 0.68, 0.72, 0.82, 0.95, 1.10]
print_loss_curve("  Overfitting:", epochs, over_train, over_val)

print("  What this means:")
print("    - The model has MEMORIZED the training data")
print("    - It learned noise and quirks, not real patterns")
print("    - Like a student who memorized every practice problem")
print("      word for word — then fails the real exam")
print()
print("  What to do:")
print("    - MORE DATA (most effective fix — always)")
print("    - Add dropout (randomly disable neurons during training)")
print("    - Data augmentation (make each image look different each epoch)")
print("    - Early stopping (stop at the lowest val loss point)")
print()

print("""
  ─────────────────────────────────────────────────────────────
  EARLY STOPPING: automatically stop at the best moment

  Your train.py already does this. Here's the logic:

    best_val_loss = infinity
    patience = 5              ← how many epochs to wait before giving up
    epochs_without_improvement = 0

    for each epoch:
        compute val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model()          ← save the weights right now
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Val loss hasn't improved in 5 epochs. Stopping.")
                break

  This means: even if you train for 100 epochs, the SAVED model
  is always from the epoch where val loss was lowest — the sweet spot.

  When you load data/shein_model.pt, you get the best version of the model,
  not the final one (which may be overfit).
""")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5: What to do when your model gets something wrong
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5: What to do when your model gets something wrong
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A wrong prediction is not just a bad number to feel sad about.
It's information. It's pointing you toward what to fix.

Here is the debugging workflow, step by step.
""")

# Synthetic predictions with image paths
random.seed(42)

# Simulate: predict category + confidence for 20 images
image_names = [f"img_{i:04d}.jpg" for i in range(20)]
actual_categories = [
    "dress", "dress", "top", "jeans", "jeans",
    "jacket", "top", "dress", "skirt", "jeans",
    "top", "jacket", "dress", "skirt", "top",
    "jeans", "jacket", "top", "dress", "skirt",
]
# Some predictions match, some don't
predicted_categories = [
    "dress", "top",    "top", "jeans", "jeans",    # img 0-4: 0,2 = top confused dress
    "jacket", "dress", "dress", "skirt", "jeans",   # img 5-9: 6 wrong
    "jacket", "jacket", "dress", "top",  "top",     # img 10-14: 10,13 wrong
    "jeans", "jacket", "top", "dress", "dress",     # img 15-19: 19 wrong
]
# Confidence scores (probability of the predicted class)
confidences = [
    0.92, 0.78, 0.95, 0.88, 0.97,
    0.91, 0.61, 0.83, 0.74, 0.96,
    0.55, 0.89, 0.71, 0.63, 0.94,
    0.99, 0.87, 0.80, 0.76, 0.52,
]

print("  Synthetic batch of 20 predictions:")
print()
print(f"  {'Image':<15} {'Actual':<10} {'Predicted':<10} {'Confidence':>12} {'Result'}")
print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*12} {'─'*7}")

wrong_predictions = []
for img, actual, predicted, conf in zip(image_names, actual_categories, predicted_categories, confidences):
    correct = actual == predicted
    result  = "OK" if correct else "WRONG"
    marker  = "  " if correct else "<<"
    print(f"  {img:<15} {actual:<10} {predicted:<10} {conf:>11.0%}  {result} {marker}")
    if not correct:
        wrong_predictions.append((img, actual, predicted, conf))

print()
print(f"  Wrong predictions: {len(wrong_predictions)} out of {len(image_names)}")
acc = (len(image_names) - len(wrong_predictions)) / len(image_names)
print(f"  Accuracy: {acc*100:.1f}%")

print("""
  ─────────────────────────────────────────────────────────────
  STEP 1: Find all wrong predictions
""")
print("  Wrong predictions only:")
print()
print(f"  {'Image':<15} {'Actual':<10} {'Predicted':<10} {'Confidence':>12} {'Danger'}")
print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*12} {'─'*30}")
for img, actual, predicted, conf in wrong_predictions:
    if conf > 0.80:
        danger = "HIGH — model is confidently wrong"
    elif conf > 0.60:
        danger = "MEDIUM — model is unsure"
    else:
        danger = "LOW — model knows it's uncertain"
    print(f"  {img:<15} {actual:<10} {predicted:<10} {conf:>11.0%}  {danger}")

print("""
  ─────────────────────────────────────────────────────────────
  STEP 2: Understand confidence — it tells you HOW wrong you are

  A wrong prediction with 99% confidence is a serious problem.
  It means the model is completely certain — and completely wrong.
  This usually means:
    - The training data for one of those classes is mislabeled
    - The two classes look nearly identical in your photos
    - One class has too few examples

  A wrong prediction with 51% confidence is much less alarming.
  The model is basically guessing — it knows something is off.
  This usually means:
    - The image is genuinely ambiguous (a wrap dress that looks like a skirt)
    - You need more training examples near this decision boundary

  ─────────────────────────────────────────────────────────────
  STEP 3: Look at the actual images

  Don't just stare at numbers. Open the misclassified images.

    wrong_dir = Path("data/errors")
    wrong_dir.mkdir(exist_ok=True)
    for img_path in wrong_image_paths:
        shutil.copy(img_path, wrong_dir / img_path.name)

  Then look at them with your eyes. You'll almost always see:
    a) The label is wrong — someone labeled a wrap skirt as a dress
    b) The photo quality is terrible — blurry, wrong background
    c) The item genuinely looks like both classes

  Each of those has a different fix.

  ─────────────────────────────────────────────────────────────
  STEP 4: Find classes with high loss

  Besides wrong predictions, look at the LOSS per class.
  High loss on a specific class = the model is very uncertain about it.

  In your training loop, you can compute this:

    for class_name in category_names:
        class_mask = (labels == class_idx[class_name])
        class_loss = loss[class_mask].mean()
        print(f"{class_name}: avg loss = {class_loss:.3f}")

  High loss classes need more training data or better labels — not more epochs.
""")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 6: Metrics for YOUR specific model
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 6: Metrics for YOUR specific model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your model is special: it predicts TWO things at once.
  - category (jacket, jeans, dress, top, skirt...)
  - color    (black, white, blue, red, navy...)

Each prediction has its own "head" — a separate output layer.
You evaluate them INDEPENDENTLY.

Why independently? Because a model can be great at category but
bad at color. You need to know which part needs work.
""")

# Simulate category and color accuracy separately
print("  Simulated evaluation results from your validation set:")
print()

category_results = {
    "jacket":  (45, 50),   # (correct, total)
    "jeans":   (89, 92),
    "dress":   (62, 77),
    "top":     (78, 92),
    "skirt":   (31, 42),
}
color_results = {
    "black":   (112, 120),
    "white":   (89,  100),
    "blue":    (74,   90),
    "red":     (41,   55),
    "navy":    (33,   48),
    "grey":    (28,   40),
}

print("  CATEGORY HEAD:")
print(f"  {'Class':<10} | {'Correct':>7} | {'Total':>7} | {'Accuracy':>9} | Visual")
print(f"  {'─'*10}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*22}")
cat_total_correct = 0
cat_total = 0
for cls, (correct, total) in category_results.items():
    acc = correct / total
    bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
    print(f"  {cls:<10} | {correct:>7} | {total:>7} | {acc:>8.1%}  | {bar}")
    cat_total_correct += correct
    cat_total += total

cat_overall = cat_total_correct / cat_total
print(f"  {'─'*10}─┴─{'─'*7}─┴─{'─'*7}─┴─{'─'*9}─┴─{'─'*22}")
print(f"  {'OVERALL':<10}   {cat_total_correct:>7}   {cat_total:>7}   {cat_overall:>8.1%}")

print()
print("  COLOR HEAD:")
print(f"  {'Color':<10} | {'Correct':>7} | {'Total':>7} | {'Accuracy':>9} | Visual")
print(f"  {'─'*10}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*22}")
col_total_correct = 0
col_total = 0
for color, (correct, total) in color_results.items():
    acc = correct / total
    bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
    print(f"  {color:<10} | {correct:>7} | {total:>7} | {acc:>8.1%}  | {bar}")
    col_total_correct += correct
    col_total += total

col_overall = col_total_correct / col_total
print(f"  {'─'*10}─┴─{'─'*7}─┴─{'─'*7}─┴─{'─'*9}─┴─{'─'*22}")
print(f"  {'OVERALL':<10}   {col_total_correct:>7}   {col_total:>7}   {col_overall:>8.1%}")

print(f"""
  ─────────────────────────────────────────────────────────────
  WHAT NUMBERS ARE REALISTIC?

  For 10+ clothing categories:
    > 85% overall category accuracy  →  good
    > 90%                            →  very good
    > 95%                            →  excellent (probably needs lots of data)

  For color:
    > 80% overall color accuracy     →  good
    > 85%                            →  very good

  Why is color harder?
    - "Navy" and "dark blue" look identical in compressed product photos
    - "Grey" and "silver" are genuinely the same color in different contexts
    - Colors look different across monitors and cameras
    - Your color at {col_overall*100:.1f}% is {'good' if col_overall > 0.80 else 'needs improvement'}

  For your specific model right now:
    Category accuracy: {cat_overall*100:.1f}%  ({'good' if cat_overall > 0.85 else 'needs improvement'})
    Color accuracy:    {col_overall*100:.1f}%  ({'good' if col_overall > 0.80 else 'needs improvement'})

  ─────────────────────────────────────────────────────────────
  READING THE TRAIN.PY VALIDATION OUTPUT

  During training, your terminal shows lines like:

    Epoch  3/20  train_loss=0.842  val_loss=0.791  cat_acc=0.723  color_acc=0.681

  Here's what each part means:
    Epoch  3/20   →  you're on the 3rd training pass through all your data
    train_loss    →  average loss on training images (you want this going down)
    val_loss      →  average loss on validation images (this is what matters)
    cat_acc       →  overall category accuracy on validation set
    color_acc     →  overall color accuracy on validation set

  If val_loss stopped decreasing 3 epochs ago: stop and look at your data.
  If cat_acc < 0.50 after 10 epochs: something is wrong with labels or data loading.
""")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 7: How to improve accuracy — the practical checklist
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 7: How to improve accuracy — the practical checklist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You've trained your model. Accuracy isn't where you want it.
Here are every lever you can pull, in order of likely impact.
""")

print("""
  ─────────────────────────────────────────────────────────────
  FIX 1: MORE DATA  (biggest impact — by far)
  ─────────────────────────────────────────────────────────────

  Why? Because every new labeled image teaches the model something
  it didn't know. This is the learning curve concept:

  As training size grows, accuracy increases but then levels off:

   100% ─
         │
    90%  │                               ●──●──●  ← levels off eventually
         │                         ●──●
    80%  │                   ●──●
         │             ●──●
    70%  │       ●──●
         │  ●──●
    60%  └──────────────────────────────────────
          100  500  1k  2k  5k  10k  50k  images

  You're in the steep part of the curve with small datasets.
  Every 1000 new labeled images is worth more than a week of hyperparameter tuning.

  For fashion specifically:
    - Scrape more product images from the web (with consistent labels)
    - Use open datasets: DeepFashion (800k images), Fashion-MNIST
    - Manually label 200 more images for your weakest class

  ─────────────────────────────────────────────────────────────
  FIX 2: BETTER AUGMENTATION
  ─────────────────────────────────────────────────────────────

  Augmentation makes each image look slightly different every epoch.
  This forces the model to learn what's essential (the clothing)
  not what's accidental (the background, lighting, zoom level).

  What you have now (in train.py):
    RandomHorizontalFlip     ← mirrors the image left/right
    ColorJitter(0.2, 0.2)    ← slight brightness/contrast changes
    RandomRotation(5)        ← tiny rotation

  What you could add:
    ColorJitter(0.4, 0.4, 0.2, 0.1)  ← more aggressive color shifts
    RandomGrayscale(p=0.1)           ← 10% chance to go grayscale
    RandomPerspective(0.2)           ← slight camera angle change
    RandomErasing(p=0.1)             ← randomly black out a patch

  Code to add in train.py transforms:
""")

print("  train_transform = transforms.Compose([")
print("      transforms.Resize(224),")
print("      transforms.RandomCrop(224, padding=16),")
print("      transforms.RandomHorizontalFlip(),")
print("      transforms.ColorJitter(brightness=0.4, contrast=0.4,")
print("                             saturation=0.2, hue=0.1),")
print("      transforms.RandomGrayscale(p=0.1),")
print("      transforms.RandomPerspective(distortion_scale=0.2, p=0.3),")
print("      transforms.ToTensor(),")
print("      transforms.Normalize(mean=..., std=...),")
print("      transforms.RandomErasing(p=0.1),")
print("  ])")

print("""
  ─────────────────────────────────────────────────────────────
  FIX 3: CLASS WEIGHTS IN CROSSENTROPYLOSS  (fix class imbalance)
  ─────────────────────────────────────────────────────────────

  Remember the jeans vs. skirts problem from Chapter 1?
  You can tell PyTorch to penalize mistakes on rare classes MORE.

  If skirts are 10x rarer than jeans, give skirt errors 10x the penalty.
  This forces the model to pay more attention to rare classes.
""")

# Show the math
category_counts = {"jacket": 800, "jeans": 2100, "dress": 950, "top": 1800, "skirt": 150}
total = sum(category_counts.values())
print("  Example category distribution:")
print()
print(f"  {'Category':<10}  {'Count':>6}  {'Frequency':>10}  {'Weight':>8}  {'Reasoning'}")
print(f"  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*30}")

for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    freq   = count / total
    weight = 1.0 / freq  # inverse frequency
    print(f"  {cat:<10}  {count:>6}  {freq:>9.1%}   {weight:>7.2f}  ← error counts {weight:.1f}x more")

print()
print("  In code (add to train.py):")
print()
print("  counts = torch.tensor([category_counts[c] for c in sorted_classes], dtype=torch.float)")
print("  weights = 1.0 / counts")
print("  weights = weights / weights.sum()   # normalize so they sum to 1")
print("  criterion = nn.CrossEntropyLoss(weight=weights.to(device))")

print("""
  ─────────────────────────────────────────────────────────────
  FIX 4: LARGER BACKBONE
  ─────────────────────────────────────────────────────────────

  You're using clip-vit-base-patch32.
  Swap it for clip-vit-large-patch14 for noticeably better features.

  Tradeoff:
    clip-vit-base-patch32  → fast, uses ~1GB VRAM, good features
    clip-vit-large-patch14 → 3x slower, uses ~3GB VRAM, better features

  Change in train.py:
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

  Expected accuracy improvement: +3% to +8% on most fashion datasets.

  ─────────────────────────────────────────────────────────────
  FIX 5: LONGER TRAINING WITH LOWER LEARNING RATE
  ─────────────────────────────────────────────────────────────

  After initial training, do a "fine-tuning pass" with 10x lower lr:
    - First training:  lr_head=1e-3, lr_backbone=1e-5
    - Fine-tune pass:  lr_head=1e-4, lr_backbone=1e-6, epochs=10

  This lets the model make small, careful adjustments without
  undoing what it already learned. Often gains +1-3%.

  ─────────────────────────────────────────────────────────────
  FIX 6: BETTER LABEL QUALITY
  ─────────────────────────────────────────────────────────────

  Mislabeled data is very common in scraped fashion datasets.
  A product tagged "dress" on a website might be a long top.
  A product tagged "navy" might display as black in the thumbnail.

  Noise in labels directly caps your accuracy. You cannot exceed
  the quality of your labels no matter how good the model is.

  To find likely mislabeled images:
    1. Train the model to ~75% accuracy
    2. Find the most confident WRONG predictions
    3. Inspect those images manually
    4. Correct the labels
    5. Retrain

  Usually 5-10% of scraped labels are wrong.
  Fixing them can improve accuracy by 3-7%.
""")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 8: Running evaluation on your trained model
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 8: Running evaluation on your trained model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now we pull everything together. Let's try to load your actual trained model
and run the full evaluation pipeline on it.
""")

MODEL_PATH = Path(__file__).parent.parent / "data" / "shein_model.pt"

print(f"  Looking for trained model at:")
print(f"  {MODEL_PATH}")
print()

if not MODEL_PATH.exists():
    print("  Model not found yet. That's fine — you haven't trained it yet.")
    print("  Run python learn/05_fine_tune.py first to generate it.")
    print()
    print("  In the meantime, here is what the REAL evaluation code looks like.")
    print("  This code will run automatically once your model exists.")
    print()
    REAL_MODEL = False
else:
    print("  Model found! Loading...")
    REAL_MODEL = True

if REAL_MODEL:
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    label_maps = checkpoint["label_maps"]

    print(f"  Saved at epoch:     {checkpoint.get('epoch', '?')}")
    print(f"  Val loss at save:   {checkpoint.get('val_loss', float('nan')):.4f}")
    print()

    # Import model class from fine_tune
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from fine_tune_05 import FashionClassifier, FashionDataset, load_data, build_label_maps
        from torchvision import transforms
        from transformers import CLIPModel, CLIPProcessor

        clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model     = FashionClassifier(label_maps, clip_base)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_size = processor.image_processor.size["shortest_edge"]

        val_transform = transforms.Compose([
            transforms.Resize(clip_size),
            transforms.CenterCrop(clip_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_processor.image_mean,
                std=processor.image_processor.image_std,
            ),
        ])

        from torch.utils.data import DataLoader
        rows = load_data()
        label_maps_real = build_label_maps(rows)
        val_rows = rows[-max(200, int(len(rows) * 0.1)):]
        val_ds = FashionDataset(val_rows, label_maps_real, val_transform)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        print("  Running evaluation pass over validation set...")
        all_preds  = defaultdict(list)
        all_labels = defaultdict(list)

        with torch.no_grad():
            for images, label_dicts in val_loader:
                preds = model(images)
                for field in label_maps_real:
                    targets = torch.tensor(
                        [label_dicts[field][i].item() for i in range(len(images))]
                    )
                    all_preds[field].extend(preds[field].argmax(1).tolist())
                    all_labels[field].extend(targets.tolist())

        print()
        for field in label_maps_real:
            idx_to_name = {v: k for k, v in label_maps_real[field].items()}
            sorted_classes = sorted(label_maps_real[field].values())

            correct_per = defaultdict(int)
            total_per   = defaultdict(int)

            for pred, true in zip(all_preds[field], all_labels[field]):
                total_per[true] += 1
                if pred == true:
                    correct_per[true] += 1

            total_correct = sum(correct_per.values())
            total_all     = sum(total_per.values())
            overall_acc   = total_correct / total_all if total_all > 0 else 0

            print(f"  PER-CLASS ACCURACY — {field.upper()}")
            print(f"  {'Class':<15} | {'Correct':>7} | {'Total':>7} | {'Accuracy':>9} | Visual")
            print(f"  {'─'*15}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*22}")
            for cls_idx in sorted_classes:
                t   = total_per[cls_idx]
                c   = correct_per[cls_idx]
                acc = c / t if t > 0 else 0
                bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
                name = idx_to_name.get(cls_idx, str(cls_idx))
                print(f"  {name:<15} | {c:>7} | {t:>7} | {acc:>8.1%}  | {bar}")

            print(f"  {'─'*15}─┴─{'─'*7}─┴─{'─'*7}─┴─{'─'*9}─┴─{'─'*22}")
            print(f"  {'OVERALL':<15}   {total_correct:>7}   {total_all:>7}   {overall_acc:>8.1%}")
            status = "GOOD" if (field == "category" and overall_acc > 0.85) or \
                               (field == "color"    and overall_acc > 0.80) else "NEEDS WORK"
            print(f"  Status: {status}")
            print()

    except ImportError as e:
        print(f"  Could not import model class: {e}")
        print("  Make sure fine_tune_05.py exists in the same directory.")

else:
    # Show what the output would look like, using our synthetic data from Chapter 6
    print("  Here is what evaluation output looks like with a trained model:")
    print()
    print("  PER-CLASS ACCURACY — CATEGORY")
    print(f"  {'Class':<15} | {'Correct':>7} | {'Total':>7} | {'Accuracy':>9} | Visual")
    print(f"  {'─'*15}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*22}")
    for cls, (correct, total) in category_results.items():
        acc = correct / total
        bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cls:<15} | {correct:>7} | {total:>7} | {acc:>8.1%}  | {bar}")
    print(f"  {'─'*15}─┴─{'─'*7}─┴─{'─'*7}─┴─{'─'*9}─┴─{'─'*22}")
    print(f"  {'OVERALL':<15}   {cat_total_correct:>7}   {cat_total:>7}   {cat_overall:>8.1%}")
    print()
    print("  PER-CLASS ACCURACY — COLOR")
    print(f"  {'Color':<15} | {'Correct':>7} | {'Total':>7} | {'Accuracy':>9} | Visual")
    print(f"  {'─'*15}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*22}")
    for color, (correct, total) in color_results.items():
        acc = correct / total
        bar = "▓" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {color:<15} | {correct:>7} | {total:>7} | {acc:>8.1%}  | {bar}")
    print(f"  {'─'*15}─┴─{'─'*7}─┴─{'─'*7}─┴─{'─'*9}─┴─{'─'*22}")
    print(f"  {'OVERALL':<15}   {col_total_correct:>7}   {col_total:>7}   {col_overall:>8.1%}")

input("\n  Press Enter to continue...")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY: What you now know
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAPTER 1 — Accuracy can lie
  Overall accuracy is almost meaningless when classes are imbalanced.
  Always look at per-class accuracy. Fashion datasets are always imbalanced.

CHAPTER 2 — Precision, Recall, F1
  Precision = when you said X, how often was it X?
  Recall    = of all real X images, how many did you find?
  F1        = the single number that balances both.
  These three together give you the full picture for each class.

CHAPTER 3 — Confusion matrix
  A grid of every (predicted, actual) pair.
  Diagonal = correct. Off-diagonal = mistakes.
  Biggest off-diagonal number = your most common error = what to fix first.

CHAPTER 4 — Overfitting vs underfitting
  Underfitting:  both losses high → model too simple / too little training
  Good fit:      both losses low and close → keep going
  Overfitting:   train loss drops, val loss rises → stop and fix data
  Early stopping: always save the model at the lowest val loss point.

CHAPTER 5 — Debugging wrong predictions
  Find all wrong predictions. Sort by confidence.
  High-confidence wrong prediction = something fundamentally wrong (mislabeled data).
  Low-confidence wrong prediction = ambiguous image, expected behavior.
  Always look at the actual images — numbers alone don't tell the whole story.

CHAPTER 6 — Your model has two heads
  Evaluate category and color accuracy independently.
  Target: >85% category accuracy, >80% color accuracy.
  Colors are genuinely harder — some ambiguity is unavoidable.

CHAPTER 7 — How to improve
  In order of impact:
    1. More data                 (biggest lever, always)
    2. Better augmentation       (easy, try it today)
    3. Class weights in loss     (fixes imbalance problem)
    4. Larger backbone           (clip-vit-large-patch14)
    5. Longer training / lower lr
    6. Fix mislabeled data       (often worth 3-7%)

CHAPTER 8 — Full evaluation pipeline
  Load model from data/shein_model.pt
  Run on validation set
  Print per-class accuracy with visual bars
  Identify weakest classes → that's your to-do list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO DO NEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  If you haven't trained yet:
    python learn/05_fine_tune.py

  Once training is done, run this file again:
    python learn/06_evaluate.py
    (it will load your real model and show you real numbers)

  Then look at your weakest class and do ONE of:
    - Collect 200 more images for that class
    - Check for mislabeled images in that class
    - Add class weights to CrossEntropyLoss for that class

  Repeat until your metrics are where you want them.

  You now have all the tools to diagnose and fix your model.
""")
print("  Lesson 6 complete.")
