"""
LESSON 2: The Training Loop — How a Model Actually Learns

Every model ever trained (GPT, CLIP, your fashion classifier) uses this same
4-step loop. We'll build up from absolute zero: what learning even means,
what weights are, what loss is, what gradients are — and finally tie it all
together into the real loop.

Run: python learn/02_training_loop.py
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1: What Does "Learning" Even Mean for a Computer?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Imagine you're teaching someone to bake a cake by adjusting a recipe.

  Day 1: They use 1 cup of sugar.  Cake is terrible.  Too bland.
  Day 2: They add 2 cups.          Cake is better.    Still off.
  Day 3: They try 1.5 cups.        Cake is great!     Perfect.

They kept tweaking one number (sugar amount) based on feedback (taste).
THAT IS EXACTLY what machine learning is.

The model starts with random numbers called "weights" — totally guessing.
After seeing each example and being told how wrong it was, it nudges
those numbers slightly in the right direction.

Do this tens of thousands of times and the weights settle into values
that produce correct answers.

There's no magic. No thinking. Just: guess → measure wrongness → adjust.
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2: What Is a Weight / Parameter?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A model is just a math function with adjustable knobs.

The simplest possible model:

    prediction = weight * x

That's it. One input (x), one number to tune (weight).

Watch what happens as we try different weights on the same input (x = 3.0),
where the true answer is 6.0:
""")

x = 3.0
true_answer = 6.0

print(f"  Input x = {x},  True answer = {true_answer}")
print()
print(f"  {'Weight':>8}  {'Prediction':>12}  {'Error':>10}  {'Comment'}")
print(f"  {'------':>8}  {'----------':>12}  {'-----':>10}  {'-------'}")

for weight in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]:
    prediction = weight * x
    error = prediction - true_answer
    comment = "WAY OFF" if abs(error) > 3 else ("CLOSE" if abs(error) < 1 else "getting there")
    print(f"  {weight:>8.1f}  {prediction:>12.2f}  {error:>+10.2f}  {comment}")

print("""
Notice: at weight=2.0, prediction = 2.0 * 3.0 = 6.0. Exactly right!

A real neural network has MILLIONS of these weights, not just one.
Your fashion classifier (CLIP) has ~150,000,000 weights.

But the principle is identical: adjust the numbers until predictions match reality.
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3: What Is Loss?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You need ONE single number to answer: "How wrong am I right now?"

That number is the LOSS.  Lower = better.  0 = perfect.

────────────────────────────────────────────────
LOSS TYPE 1: Mean Squared Error (for regression)
────────────────────────────────────────────────
Used when you're predicting a continuous value (e.g. "what price is this item?")

    MSE = (prediction - truth)²

Why squared? Two reasons:
  1. Makes negatives positive (you don't want errors to cancel out)
  2. Punishes big errors MORE than small ones (2² = 4, but 10² = 100)

Let's compute it by hand:
""")

examples = [
    (0.1 * 3.0, 6.0),  # weight=0.1
    (0.5 * 3.0, 6.0),  # weight=0.5
    (1.0 * 3.0, 6.0),  # weight=1.0
    (1.5 * 3.0, 6.0),  # weight=1.5
    (2.0 * 3.0, 6.0),  # weight=2.0
]

print(f"  {'Weight':>8}  {'Prediction':>12}  {'(pred-truth)²':>15}  {'= MSE'}")
print(f"  {'------':>8}  {'----------':>12}  {'-------------':>15}  {'-----'}")
for w, (pred, truth) in zip([0.1, 0.5, 1.0, 1.5, 2.0], examples):
    mse = (pred - truth) ** 2
    diff = pred - truth
    print(f"  {w:>8.1f}  {pred:>12.2f}  ({pred:.2f}-{truth:.2f})²={diff:.2f}²  = {mse:.4f}")

print("""
At weight=2.0:  prediction=6.0, truth=6.0  →  (6.0-6.0)² = 0.0  ← PERFECT!

────────────────────────────────────────────────────────
LOSS TYPE 2: Cross-Entropy Loss (for classification)
────────────────────────────────────────────────────────
Used when you're predicting a CATEGORY (e.g. "is this a jacket or jeans?")

Your model outputs a probability for each class:
  e.g.  { jacket: 0.70,  jeans: 0.20,  top: 0.10 }

Cross-Entropy Loss measures how surprised the model was by the true label.
  - If true label = jacket and model said 0.70 → low loss (model was confident + right)
  - If true label = jacket and model said 0.05 → high loss (model was confident + wrong)

The formula:  Loss = -log( probability assigned to the correct class )
""")

import math

print("  Probability model gave to correct class  →  Loss  →  Interpretation")
print("  " + "-" * 65)
for prob in [0.95, 0.70, 0.50, 0.20, 0.05]:
    loss = -math.log(prob)
    interp = ("Excellent" if prob > 0.8 else
              "Good" if prob > 0.5 else
              "Mediocre" if prob > 0.25 else
              "Very bad")
    print(f"  {prob:.2f}  ({prob*100:.0f}% confident, correct)       →  {loss:.3f}  →  {interp}")

print("""
So if the model outputs 95% confidence and it's correct: loss = 0.051.  Nearly 0.
If it outputs 5% confidence and it's correct:            loss = 2.996.  High penalty.

In PyTorch this is: nn.CrossEntropyLoss()
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4: What Is a Gradient?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You know you need to reduce the loss. But which direction should you move the weight?

Imagine you're blindfolded on a hilly landscape. Your feet are on the slope.
You want to get to the lowest point (the valley = lowest loss).

The GRADIENT tells you: "The hill slopes upward in THIS direction."
So you step the OPPOSITE way.

Let's see this concretely. We'll compute loss at several nearby weight values,
and watch the loss go down, then up (forming a U-shape):
""")

x_val = 3.0
truth = 6.0

print(f"  (Predicting {truth:.1f} from x={x_val:.1f}, so ideal weight = {truth/x_val:.1f})")
print()
print(f"  {'Weight':>8}  {'Prediction':>12}  {'MSE Loss':>10}  {'Trend'}")
print(f"  {'------':>8}  {'----------':>12}  {'--------':>10}  {'-----'}")

prev_loss = None
for w_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    pred = w_val * x_val
    loss_val = (pred - truth) ** 2
    trend = ""
    if prev_loss is not None:
        trend = "↓ decreasing" if loss_val < prev_loss else "↑ increasing"
    print(f"  {w_val:>8.2f}  {pred:>12.2f}  {loss_val:>10.4f}  {trend}")
    prev_loss = loss_val

print("""
Loss is LOWEST at weight=2.0 and rises in both directions.

The GRADIENT is the slope of this curve at your current weight:
  - If weight=1.0 (left of valley): gradient is NEGATIVE → go right (increase weight)
  - If weight=3.0 (right of valley): gradient is POSITIVE → go left (decrease weight)
  - If weight=2.0 (bottom of valley): gradient is ZERO → you're done!

This is exactly what PyTorch's loss.backward() computes — automatically.
It figures out the slope (gradient) for EVERY weight in the model simultaneously.

Let's watch PyTorch compute a gradient:
""")

weight_t = torch.tensor([1.5], requires_grad=True)  # requires_grad=True = "track this"
x_t = torch.tensor([3.0])
truth_t = torch.tensor([6.0])

prediction_t = weight_t * x_t
loss_t = (prediction_t - truth_t) ** 2

print(f"  weight          = {weight_t.item():.2f}")
print(f"  prediction      = weight * x = {weight_t.item():.2f} * {x_t.item():.1f} = {prediction_t.item():.2f}")
print(f"  loss (MSE)      = ({prediction_t.item():.2f} - {truth_t.item():.1f})² = {loss_t.item():.4f}")
print()

loss_t.backward()  # PyTorch computes the gradient here

print(f"  gradient (d_loss/d_weight) = {weight_t.grad.item():.4f}")
print(f"""
  Interpretation: For every +1 increase in weight, the loss changes by {weight_t.grad.item():.4f}.
  Since the gradient is NEGATIVE, increasing weight will DECREASE loss.
  → We should increase our weight to get closer to 2.0.

  Manual check: d/dw [(w*x - truth)²] = 2*(w*x - truth)*x
  = 2 * ({weight_t.item():.2f}*{x_t.item():.1f} - {truth_t.item():.1f}) * {x_t.item():.1f}
  = 2 * ({prediction_t.item():.2f} - {truth_t.item():.1f}) * {x_t.item():.1f}
  = 2 * ({prediction_t.item() - truth_t.item():.2f}) * {x_t.item():.1f}
  = {2 * (prediction_t.item() - truth_t.item()) * x_t.item():.4f}   ← matches .grad above!
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5: What Is an Optimizer?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You now know the gradient = the slope.
The OPTIMIZER uses the gradient to update the weight.

The simplest rule:   new_weight = old_weight - learning_rate * gradient

"Learning rate" is how big a step to take.

──────────────────────────────────────────────────────
WHY NOT TAKE GIANT STEPS AND LEARN FASTER?
──────────────────────────────────────────────────────
Imagine you're on that hilly landscape again.
  - Too big a step: you leap OVER the valley to the other side. Loss gets worse.
  - Too small a step: you inch forward. Takes 10,000 steps to get there.
  - Just right: steady progress, arrives at the bottom efficiently.

Let's watch manual gradient descent converge:
""")

w = torch.tensor([0.5], requires_grad=True)
x_t2 = torch.tensor([3.0])
truth_t2 = torch.tensor([6.0])
lr = 0.05

print(f"  Starting weight: {w.item():.4f}   Target: {truth_t2.item()/x_t2.item():.4f}")
print(f"  Learning rate:   {lr}")
print()
print(f"  {'Step':>6}  {'Weight':>10}  {'Prediction':>12}  {'Loss':>10}  {'Gradient':>12}")
print(f"  {'----':>6}  {'------':>10}  {'----------':>12}  {'----':>10}  {'--------':>12}")

for step in range(1, 16):
    pred = w * x_t2
    loss_manual = (pred - truth_t2) ** 2
    loss_manual.backward()

    with torch.no_grad():
        grad_val = w.grad.item()
        w_val = w.item()
        pred_val = pred.item()
        loss_val = loss_manual.item()

        # Manual update
        w -= lr * w.grad
        w.grad.zero_()  # MUST clear gradient or it accumulates next step

    if step <= 8 or step == 15:
        note = "  ← converged!" if step == 15 else ""
        print(f"  {step:>6}  {w_val:>10.4f}  {pred_val:>12.4f}  {loss_val:>10.6f}  {grad_val:>12.4f}{note}")
    elif step == 9:
        print(f"  {'...':>6}")

print(f"""
After 15 steps, weight ≈ {w.item():.4f}  (target was 2.0000).

Now let's see torch.optim.Adam do the same job automatically:
""")

weight_adam = torch.tensor([0.5], requires_grad=True)
optimizer_adam = torch.optim.Adam([weight_adam], lr=0.1)

print(f"  {'Step':>6}  {'Weight':>10}  {'Loss':>10}  {'Method'}")
print(f"  {'----':>6}  {'------':>10}  {'----':>10}  {'------'}")

for step in range(1, 21):
    pred = weight_adam * x_t2
    loss_a = (pred - truth_t2) ** 2

    optimizer_adam.zero_grad()   # Step A: clear old gradients
    loss_a.backward()            # Step B: compute new gradients
    optimizer_adam.step()        # Step C: update weight

    if step in [1, 5, 10, 15, 20]:
        print(f"  {step:>6}  {weight_adam.item():>10.4f}  {loss_a.item():>10.6f}  Adam")

print(f"""
Adam converges faster than plain gradient descent because it adapts
the learning rate for each parameter individually.

In your fashion model, the optimizer is:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

It's updating ALL ~150 million weights at once with one call to optimizer.step().
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 6: The Full 4-Step Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now we tie EVERYTHING together.

The 4 steps — memorize these, they appear in every training script:

    Step 1  (Forward)   predictions = model(inputs)
    Step 2  (Loss)      loss = criterion(predictions, labels)
    Step 3  (Backward)  optimizer.zero_grad(); loss.backward()
    Step 4  (Update)    optimizer.step()

We'll train a tiny model on a real toy problem:
  Input:  (x, y) coordinates — two numbers
  Task:   predict whether x + y > 1.0  (class 1) or ≤ 1.0  (class 0)
  Why:    simple enough to see clearly, but non-trivial enough to actually learn
""")

# ── Data ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
X = torch.rand(200, 2)
Y = ((X[:, 0] + X[:, 1]) > 1.0).long()

n_class0 = (Y == 0).sum().item()
n_class1 = (Y == 1).sum().item()

print(f"  Dataset: 200 points in 2D space")
print(f"  Class 0 (x+y ≤ 1.0): {n_class0} points")
print(f"  Class 1 (x+y > 1.0): {n_class1} points")
print()
print(f"  First 5 samples:")
print(f"  {'x':>8}  {'y':>8}  {'x+y':>8}  {'label'}")
print(f"  {'---':>8}  {'---':>8}  {'---':>8}  {'-----'}")
for i in range(5):
    xi, yi = X[i, 0].item(), X[i, 1].item()
    label = Y[i].item()
    print(f"  {xi:>8.3f}  {yi:>8.3f}  {xi+yi:>8.3f}  {label}  ({'above' if label else 'below'} the line)")

# ── Model ─────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 16),   # 2 inputs → 16 hidden neurons
    nn.ReLU(),          # ReLU(x) = max(0,x)  — lets the model learn curves
    nn.Linear(16, 2),   # 16 hidden → 2 outputs (one score per class)
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model: 2 → 16 → ReLU → 2")
print(f"  Total parameters: {total_params}")

# Layer math:
#   Linear(2, 16): weights = 2*16 = 32, biases = 16  → 48
#   Linear(16, 2): weights = 16*2 = 32, biases = 2   → 34
#   Total = 82

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# Random predictions before training
with torch.no_grad():
    raw_preds = model(X)
    initial_acc = (raw_preds.argmax(dim=1) == Y).float().mean().item()

print(f"\n  Accuracy BEFORE training (random weights): {initial_acc:.1%}")
print(f"  (Expected ~50% — model is just guessing)")

# ── Training ──────────────────────────────────────────────────────────────────
print(f"""
  Starting training...
  Watching loss drop and accuracy rise:

  {'Epoch':>7}  {'Loss':>10}  {'Accuracy':>10}  {'Notes'}""")
print(f"  {'-----':>7}  {'----':>10}  {'--------':>10}  {'-----'}")

for epoch in range(60):

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: FORWARD PASS
    # Run every input through the model to get predictions.
    # "predictions" shape: [200, 2] — two scores per sample
    # ──────────────────────────────────────────────────────────────────────
    predictions = model(X)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: COMPUTE LOSS
    # CrossEntropyLoss compares the predictions to the true labels.
    # Returns a single number: how wrong we are on average right now.
    # ──────────────────────────────────────────────────────────────────────
    loss = criterion(predictions, Y)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: BACKWARD PASS
    # zero_grad() MUST come first — PyTorch adds gradients to whatever was
    # already stored, so leftover gradients from last step would corrupt this one.
    # loss.backward() computes d(loss)/d(weight) for every weight in the model.
    # ──────────────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: UPDATE WEIGHTS
    # The optimizer reads every .grad and nudges each weight slightly.
    # After this call, all 82 weights have been updated.
    # ──────────────────────────────────────────────────────────────────────
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            acc = (predictions.argmax(dim=1) == Y).float().mean().item()
        note = ""
        if epoch + 1 == 10:  note = "← still learning"
        if epoch + 1 == 30:  note = "← getting there"
        if epoch + 1 == 60:  note = "← converged!"
        print(f"  {epoch+1:>7}  {loss.item():>10.4f}  {acc:>10.1%}  {note}")

# Final evaluation
model.eval()
with torch.no_grad():
    final_preds = model(X)
    final_acc = (final_preds.argmax(dim=1) == Y).float().mean().item()

print(f"""
  Final accuracy: {final_acc:.1%}

  Test on specific points:
""")

test_cases = [
    (torch.tensor([[0.2, 0.2]]), "below line", 0),
    (torch.tensor([[0.8, 0.8]]), "above line", 1),
    (torch.tensor([[0.6, 0.5]]), "above line", 1),
    (torch.tensor([[0.1, 0.3]]), "below line", 0),
]

print(f"  {'Point':>18}  {'True Class':>12}  {'Predicted':>10}  {'Confidence':>12}  {'Correct?'}")
print(f"  {'-----':>18}  {'----------':>12}  {'---------':>10}  {'----------':>12}  {'--------'}")

with torch.no_grad():
    for pt, desc, true_label in test_cases:
        logits = model(pt)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        correct = "YES" if pred_class == true_label else "NO"
        print(f"  ({pt[0,0].item():.1f}, {pt[0,1].item():.1f}) {desc:>12}  {'class '+str(true_label):>12}  {'class '+str(pred_class):>10}  {confidence:>11.1%}  {correct}")

input("\n  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 7: How This Applies to Your Fashion Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Everything you just watched happening on toy (x,y) points is EXACTLY
what happens when training on fashion images.

The only differences are:

┌──────────────────────┬──────────────────────────┬────────────────────────────────┐
│  Concept             │  Toy example (above)     │  Fashion classifier (train.py) │
├──────────────────────┼──────────────────────────┼────────────────────────────────┤
│  Input               │  (x, y) → 2 numbers      │  224×224 image → 150,528 pixels│
│  Model               │  2 → 16 → 2              │  CLIP visual encoder (huge)    │
│  Parameters          │  82                      │  ~150 million                  │
│  Classes             │  2 (above/below line)    │  N fashion categories          │
│  Loss function       │  CrossEntropyLoss        │  CrossEntropyLoss  (same!)     │
│  Optimizer           │  Adam                    │  Adam              (same!)     │
│  The 4 steps         │  identical               │  identical                     │
└──────────────────────┴──────────────────────────┴────────────────────────────────┘

The 4 steps in train.py look like this:

    for images, labels in train_loader:        # load one batch
        preds = model(images)                  # Step 1: forward pass
        loss  = criterion(preds, labels)       # Step 2: compute loss
        optimizer.zero_grad()                  # Step 3a: clear old gradients
        loss.backward()                        # Step 3b: compute new gradients
        optimizer.step()                       # Step 4: update weights

That's it. The model is bigger. The data is images. The math is deeper.
But the loop is the same four lines you just watched work above.

One more thing you'll see in train.py: model.train() and model.eval()
  - model.train()  → enables dropout (random neuron shutoff during training)
  - model.eval()   → disables dropout (consistent outputs during evaluation)
  Always switch between them at the right time or your results will be wrong.
""")
input("  Press Enter to continue...")

# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU LEARNED:

  Learning     = adjusting weights based on feedback, over and over
  Weight       = a tunable number the model multiplies inputs by
  Loss         = one number capturing "how wrong am I right now?"
                 MSE for regression,  CrossEntropy for classification
  Gradient     = slope of the loss landscape at your current weights
                 Tells you which direction to move each weight
  Backward()   = PyTorch computing all gradients automatically
  Optimizer    = uses gradients to update every weight (step 4)
  Learning rate = how large each update step is

THE 4 STEPS (burned into your memory forever):

    predictions = model(inputs)                # forward
    loss        = criterion(predictions, labels) # measure error
    optimizer.zero_grad(); loss.backward()     # compute gradients
    optimizer.step()                           # update weights

These 4 lines run in a loop for thousands of iterations.
Every major ML model in history was trained with this loop.

Next: learn/03_dataset.py  →  How to feed real images into this loop
""")

print("  Lesson 2 complete.  Run: python learn/03_dataset.py")
