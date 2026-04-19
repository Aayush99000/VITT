# Siamese Neural Networks for Satellite Image Change Detection
## A Complete Research Booklet: U-Net · Vision Transformer · Swin Transformer

**Dataset:** LEVIR-CD (Large-scale Remote Sensing Change Detection)  
**Institution:** Northeastern University  
**Models trained from scratch — no pretrained weights**

---

## Table of Contents

1. [Change Detection: Problem Definition](#1-change-detection-problem-definition)
2. [Dataset: LEVIR-CD](#2-dataset-levir-cd)
3. [The Siamese Principle](#3-the-siamese-principle)
4. [Vision Transformer (ViT) — Complete Architecture](#4-vision-transformer-vit--complete-architecture)
5. [Siamese ViT for Change Detection](#5-siamese-vit-for-change-detection)
6. [Siamese U-Net — Complete Architecture](#6-siamese-u-net--complete-architecture)
7. [Swin Transformer — Complete Architecture](#7-swin-transformer--complete-architecture)
8. [Feature Difference Module](#8-feature-difference-module)
9. [Progressive Decoder](#9-progressive-decoder)
10. [Loss Functions](#10-loss-functions)
11. [Training Pipeline](#11-training-pipeline)
12. [Data Augmentation](#12-data-augmentation)
13. [Experimental Results](#13-experimental-results)
14. [Analysis: Why U-Net Outperforms ViT](#14-analysis-why-u-net-outperforms-vit)
15. [Quiz: 30 Questions with Answers](#15-quiz-30-questions-with-answers)

---

## 1. Change Detection: Problem Definition

### 1.1 What Is Change Detection?

Change detection is the process of identifying differences between two images of the same geographic location captured at different points in time (T1 = before, T2 = after). In the remote sensing domain, this is a **binary pixel-level segmentation** task:

```
Given: img_T1 ∈ ℝ^(H×W×C)  and  img_T2 ∈ ℝ^(H×W×C)
Find:  mask  ∈ {0,1}^(H×W)

where mask(i,j) = 1  if pixel (i,j) changed between T1 and T2
                = 0  otherwise
```

### 1.2 Real-World Applications

| Application | What is Detected |
|---|---|
| Urban growth monitoring | New buildings, roads, parking lots |
| Disaster assessment | Collapsed structures, flood extent |
| Illegal construction | Unauthorised buildings |
| Deforestation | Forest cleared for agriculture |
| Military intelligence | New infrastructure, vehicle movement |

### 1.3 Why Is Change Detection Hard?

**Challenge 1 — Class Imbalance:**  
In LEVIR-CD, changed pixels represent less than 10% of the total. A model that predicts all zeros achieves >90% accuracy but has F1 = 0. Standard binary cross-entropy loss is blind to this problem.

**Challenge 2 — Bi-temporal consistency:**  
The model must compare two images captured under potentially different lighting, weather, and sensor conditions. The comparison must be robust to these nuisance factors while remaining sensitive to genuine structural change.

**Challenge 3 — Multi-scale structure:**  
A changed region might be a single rooftop (small) or an entire city block (large). The model must capture both fine-grained edges and coarse semantic structure simultaneously.

**Challenge 4 — Limited labelled data:**  
LEVIR-CD contains only 411 training image pairs. Deep models with tens of millions of parameters risk overfitting on such small datasets.

### 1.4 Evaluation Metrics

**Precision** — of all pixels we called "changed", what fraction truly changed?
```
Precision = TP / (TP + FP)
```

**Recall** — of all pixels that actually changed, what fraction did we detect?
```
Recall = TP / (TP + FN)
```

**F1 Score** — harmonic mean of precision and recall (primary metric):
```
F1 = 2 · Precision · Recall / (Precision + Recall)
   = 2·TP / (2·TP + FP + FN)
```

**Intersection over Union (IoU / Jaccard Index):**
```
IoU = TP / (TP + FP + FN)
```

**Cohen's Kappa** — measures agreement beyond chance:
```
κ = (p_o − p_e) / (1 − p_e)

where p_o = observed accuracy
      p_e = expected accuracy by chance
```
Kappa > 0.8 is considered "very strong agreement."

---

## 2. Dataset: LEVIR-CD

### 2.1 Overview

LEVIR-CD (Large-scale Change Detection) is a publicly available benchmark dataset of high-resolution Google Earth satellite images focused on building-level change detection in urban areas.

| Property | Value |
|---|---|
| Image size | 1024 × 1024 pixels |
| Channels | RGB (3 channels) |
| Label format | Binary mask: 0 = no change, 255 = change |
| Change type | Building construction / demolition |
| Source | Google Earth |

### 2.2 Dataset Splits

| Split | Image Pairs | Usage |
|---|---|---|
| Train | 411 | Model training |
| Val | 57 | Hyperparameter selection, early stopping |
| Test | 64 | Final evaluation |

### 2.3 Directory Structure

```
LEVIR CD/
├── train/
│   ├── A/        ← Before (T1) images: 411 × PNG (1024×1024, RGB)
│   ├── B/        ← After  (T2) images: 411 × PNG (1024×1024, RGB)
│   └── label/    ← Binary masks:       411 × PNG (1024×1024, grayscale)
├── val/
│   ├── A/, B/, label/   (57 pairs)
└── test/
    ├── A/, B/, label/   (64 pairs)
```

### 2.4 Data Loading Pipeline

```python
# Pseudocode for OSCDDataset.__getitem__

img1 = PIL.open(A / stem.png).convert("RGB")    # (1024, 1024, 3) uint8
img2 = PIL.open(B / stem.png).convert("RGB")    # (1024, 1024, 3) uint8
mask = PIL.open(label / stem.png).convert("L")  # (1024, 1024) uint8

# Binarize: 255 → 1, 0 → 0
mask = (mask == 255).astype(uint8)

# Apply augmentation (both images and mask, spatially synchronized)
result = transform(image=img1, image2=img2, mask=mask)

# Convert to tensors
image1_tensor = (H,W,3) → (3,H,W) float32, normalized
image2_tensor = (H,W,3) → (3,H,W) float32, normalized
mask_tensor   = (H,W)   → (1,H,W) float32, values in {0.0, 1.0}
```

### 2.5 Multi-Crop Augmentation

Each image stem is virtually replicated `n_crops` times per epoch. Each call draws a fresh random crop, providing `n_crops` distinct 256×256 windows from the 1024×1024 original.

```
Effective training set size = 411 stems × n_crops = 411 × 4 = 1,644 patches/epoch
Steps per epoch (batch_size=8) = 1,644 / 8 ≈ 190
```

---

## 3. The Siamese Principle

### 3.1 Why "Siamese"?

A Siamese network uses two identical sub-networks with **shared weights** to process two inputs in parallel. The outputs live in the same representation space, making direct comparison meaningful.

```
img_T1  →  Encoder(θ)  →  feats_T1
img_T2  →  Encoder(θ)  →  feats_T2   ← SAME θ (shared weights)

change_representation = f(feats_T1, feats_T2)
```

### 3.2 Why Shared Weights Matter

If the two encoders had different weights:

- Feature space of T1 ≠ feature space of T2
- `feats_T1 - feats_T2` would be meaningless (like subtracting apples from oranges)
- The model would need to separately learn alignment

With shared weights:
- Same input → same output (equivariance)
- Differences in features directly reflect differences in images
- Half the parameters to train

### 3.3 The Siamese Constraint During Training

Both images pass through the same forward pass. Gradients from both paths flow back into the same set of parameters, effectively doubling the gradient signal for the encoder at each step.

---

## 4. Vision Transformer (ViT) — Complete Architecture

### 4.1 Motivation: From Language to Vision

The original Transformer (Vaswani et al., 2017) was designed for natural language processing. ViT (Dosovitskiy et al., 2020) adapts it directly to images by treating image patches as tokens analogous to words.

**Core idea:**
```
"The cat sat on the mat."
→ [The] [cat] [sat] [on] [the] [mat]   ← word tokens

A 256×256 image with 16×16 patches:
→ 256 patch tokens, each representing a 16×16 region
```

### 4.2 Patch Embedding

**Input:** image `x` ∈ ℝ^(B × 3 × 256 × 256)

**Step 1 — Divide into patches:**

```
Number of patches: N = (H/P) × (W/P) = (256/16) × (256/16) = 16 × 16 = 256
Each patch: 16 × 16 × 3 = 768 values
```

**Step 2 — Linear projection (implemented as Conv2d):**

```
Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)

(B, 3, 256, 256) → (B, 768, 16, 16) → flatten spatial → (B, 256, 768)
```

This is mathematically equivalent to: for each patch `p_i` ∈ ℝ^768, compute `z_i = E · p_i` where `E` ∈ ℝ^(768×768) is the learnable projection matrix.

**Step 3 — Prepend [CLS] token:**

```
x_cls ∈ ℝ^768  (learnable parameter, randomly initialized)

(B, 256, 768)  →  cat([x_cls, patches], dim=1)  →  (B, 257, 768)
```

The CLS token aggregates global image-level information through the transformer.

**Step 4 — Add positional embeddings:**

```
E_pos ∈ ℝ^(257 × 768)  (learnable, initialized: truncated normal σ=0.02)

x = x + E_pos    (element-wise addition, broadcast over batch)
```

**Why positional embeddings?**  
Transformers are **permutation-invariant** — without positional information, the model cannot distinguish patch at position (0,0) from patch at position (15,15). Positional embeddings break this symmetry by giving each token a unique spatial identity.

**Step 5 — Dropout(p=0.1)**

**Output:** `x` ∈ ℝ^(B × 257 × 768)

**Parameter count:**
```
Conv2d weight:         3 × 768 × 16 × 16  =  589,824
Conv2d bias:           768
CLS token:             768
Positional embeddings: 257 × 768          =  197,376
Total:                                     ≈  789,736
```

### 4.3 Transformer Encoder Block

The ViT encoder consists of `depth = 12` identical transformer blocks stacked sequentially. Each block uses the **Pre-LayerNorm** (Pre-LN) design.

**Full block computation:**

```
Input:  x  ∈ ℝ^(B × N × d)     where N=257, d=768

x' = x + Dropout(MHSA(LayerNorm(x)))     ← attention sub-layer + residual
y  = x' + Dropout(MLP(LayerNorm(x')))    ← feedforward sub-layer + residual

Output: y  ∈ ℝ^(B × N × d)
```

**Why Pre-LN over Post-LN?**  
Post-LN (original Transformer): `x' = LayerNorm(x + MHSA(x))`  
The gradient must pass through the LayerNorm before reaching the residual connection, causing gradient instability in deep networks.  
Pre-LN: `x' = x + MHSA(LayerNorm(x))`  
The residual path is "clean" — gradients flow unimpeded through the addition, enabling stable training of 12 deep blocks from scratch.

### 4.4 Layer Normalisation

```
LN(x) = γ ⊙ (x − μ) / √(σ² + ε)  +  β

where:
  μ    = mean over the last dimension (d=768), computed per token per sample
  σ²   = variance over the last dimension
  ε    = 1×10⁻⁵  (numerical stability)
  γ, β ∈ ℝ^768   (learnable scale and shift, per-dimension)
```

**LayerNorm vs BatchNorm:**  
BatchNorm normalises over the batch dimension — requires large batches and behaves differently at train/inference time. LayerNorm normalises over the feature dimension — batch-size independent, same behaviour at train and inference. This makes LayerNorm the standard choice for transformers.

### 4.5 Multi-Head Self-Attention (MHSA)

This is the defining operation of the transformer. Every token attends to every other token simultaneously.

**Inputs:** token matrix `X` ∈ ℝ^(N × d_model) where N=257, d_model=768

**Step 1 — Linear projections:**

```
Q = X · W_Q + b_Q ∈ ℝ^(N × d_model)    W_Q ∈ ℝ^(768×768)
K = X · W_K + b_K ∈ ℝ^(N × d_model)    W_K ∈ ℝ^(768×768)
V = X · W_V + b_V ∈ ℝ^(N × d_model)    W_V ∈ ℝ^(768×768)
```

- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What do I output if selected?"

**Step 2 — Split into h=12 heads:**

```
d_k = d_model / h = 768 / 12 = 64    (dimension per head)

Q_i, K_i, V_i ∈ ℝ^(N × 64)    for i = 1, 2, ..., 12
```

**Step 3 — Scaled Dot-Product Attention per head:**

```
                     ⎛  Q_i · K_i^T  ⎞
Attention(Q_i,K_i,V_i) = softmax ⎜ ─────────────  ⎟ · V_i
                     ⎝     √d_k      ⎠

= softmax(Q_i · K_i^T / 8) · V_i
```

The matrix `Q_i · K_i^T` ∈ ℝ^(N×N) is the **attention score matrix** — entry (i,j) measures how much token i should attend to token j.

**Why divide by √d_k = √64 = 8?**  
As d_k grows, the dot products grow in magnitude (variance scales with d_k). Large values push softmax into regions with near-zero gradients. Dividing by √d_k keeps variance at 1, maintaining healthy gradient flow.

**Step 4 — Concatenate heads and project:**

```
MultiHead(Q,K,V) = [head_1 ; head_2 ; ... ; head_12] · W_O

where head_i = Attention(Q_i, K_i, V_i) ∈ ℝ^(N × 64)
concat result ∈ ℝ^(N × 768)
W_O ∈ ℝ^(768×768)
```

**Why multiple heads?**  
Different heads can specialise in different types of relationships:
- Head 1 → spatial proximity between patches
- Head 4 → colour/texture similarity
- Head 8 → semantic category correspondence
- Head 11 → long-range structural dependencies

The final projection `W_O` learns to combine all these relationship types.

**MHSA parameters per block:**
```
W_Q: 768 × 768 + 768  =  590,592
W_K: 768 × 768 + 768  =  590,592
W_V: 768 × 768 + 768  =  590,592
W_O: 768 × 768 + 768  =  590,592
Total per block:          2,362,368
```

**Computational complexity:**
```
O(N² · d)  per layer
= O(257² × 768)
= O(50,692,608)  per block per batch element
```
This quadratic dependence on sequence length N is why pure ViT is expensive at high resolution.

### 4.6 MLP Feedforward Block

After attention mixes information across tokens globally, the MLP transforms each token independently (position-wise), adding non-linear capacity.

```
Input: x ∈ ℝ^(N × 768)

x → Linear(768 → 3072) → GELU → Dropout(0.1) → Linear(3072 → 768) → Dropout(0.1)

Output: ℝ^(N × 768)
```

**The expansion ratio is 4:** `3072 = 4 × 768`. This is the standard ViT-Base configuration — the hidden layer is 4× wider than the embedding dimension.

**GELU Activation:**

```
GELU(x) = x · Φ(x)

where Φ(x) = CDF of the standard normal distribution
           = 0.5 · (1 + erf(x / √2))

Approximation:
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

**GELU vs ReLU:**  
ReLU = max(0, x): hard gate, zero gradient for x < 0 (dead neurons).  
GELU: smooth, stochastic gate. Non-zero output for slightly negative inputs. Standard across all modern transformers (GPT, BERT, ViT, Swin).

**MLP parameters per block:**
```
fc1: 768 × 3072 + 3072  =  2,362,368
fc2: 3072 × 768 + 768   =  2,362,368
Total per block:           4,724,736
```

### 4.7 Complete ViT Encoder Summary

**Total parameters:**
```
Patch embedding:           789,736
12 × TransformerBlock:  85,047,936  (each block: 7,087,104)
  ├─ MHSA:    2,362,368 × 12  =  28,348,416
  ├─ MLP:     4,724,736 × 12  =  56,696,832
  └─ LayerNorm: 3,072 × 12    =      36,864
Final LayerNorm:               1,536
Total encoder:              85,844,736
```

**Dimension flow through encoder:**
```
(B, 3, 256, 256)          Input image pair
    ↓ PatchEmbedding
(B, 257, 768)              257 tokens (256 patches + CLS), each 768-dim
    ↓ Block 1
(B, 257, 768)              Shape unchanged; content transformed
    ↓ Block 2
    ↓ ...
    ↓ Block 12
(B, 257, 768)              Final encoder output
```

### 4.8 Multi-Scale Feature Tapping

Rather than using only the final block's output, we tap features at 4 intermediate depths to capture representations at different abstraction levels.

```
Tap indices (0-based): [depth//4 - 1, depth//2 - 1, 3*depth//4 - 1, depth-1]
                     = [2, 5, 8, 11]   for depth=12

Scale 0 → after block 2:  shallow — local texture, edges, patch boundaries
Scale 1 → after block 5:  mid     — structural parts, colour blobs
Scale 2 → after block 8:  deep    — semantic shapes, building components
Scale 3 → after block 11: deepest — full global context, object-level semantics
```

Each scale output has shape `(B, 256, 768)` (CLS token dropped, only patch tokens kept).

Each scale has a **dedicated LayerNorm** to independently normalise its representation:
- Scale 0, 1, 2: `scale_norms[0]`, `scale_norms[1]`, `scale_norms[2]`
- Scale 3 (deepest): reuses `self.norm` (the standard final LN)

This design ensures `self.norm` always receives gradients (avoiding dead parameters).

---

## 5. Siamese ViT for Change Detection

### 5.1 Architecture Flow

```
img_T1 (B, 3, 256, 256) ──┐
                            ├─→ SiameseViTEncoder(θ) ─→ feats_T1  [4 × (B, 256, 768)]
img_T2 (B, 3, 256, 256) ──┘                         └─→ feats_T2  [4 × (B, 256, 768)]
                                                               │
                                              MultiScaleDiffModule
                                              (diff at each scale → fuse)
                                                               │
                                                 diff_feat (B, 256, 256)
                                                               │
                                               ProgressiveDecoder
                                               (token space → pixel space)
                                                               │
                                           change_logits (B, 1, 256, 256)
```

### 5.2 Parameter Breakdown

| Component | Parameters | % of Total |
|---|---|---|
| SiameseViTEncoder | 85,844,736 | 96.2% |
| MultiScaleDiffModule | 2,556,928 | 2.9% |
| ProgressiveDecoder | 1,015,473 | 1.1% |
| **Total** | **89,417,137** | |

### 5.3 Forward Pass (inference)

```python
feats_T1, feats_T2 = encoder(img_T1, img_T2)   # shared encoder, two passes
diff_feat           = diff_module(feats_T1, feats_T2)
change_logits       = decoder(diff_feat)
change_probs        = sigmoid(change_logits)
change_mask         = (change_probs >= threshold).float()   # threshold = 0.35
```

---

## 6. Siamese U-Net — Complete Architecture

### 6.1 Overview

The Siamese U-Net implements the FC-Siam-diff architecture. It is a **CNN-based** model with a symmetric encoder-decoder structure. The key change detection mechanism: take the **absolute feature difference** at each encoder skip connection.

```
img_T1 (B, 3, 256, 256) ──┐
                            ├─→ Shared CNN Encoder
img_T2 (B, 3, 256, 256) ──┘
                               ↓ (both images processed independently)

Skip diffs: |enc_T1_i − enc_T2_i|  at each resolution level

Bottleneck diff: |bottleneck_T1 − bottleneck_T2|
                               ↓
                           Decoder
                      (with skip diffs)
                               ↓
                   (B, 1, 256, 256) logits
```

### 6.2 Encoder Architecture

Each encoder stage consists of **two Conv-BN-ReLU blocks** followed by MaxPool:

```
ENCODER STAGE TEMPLATE:
    Input (B, C_in, H, W)
        │
        ├─→ Conv2d(C_in, C_out, kernel=3, padding=1, bias=False)
        ├─→ BatchNorm2d(C_out)
        ├─→ ReLU(inplace=True)
        ├─→ Conv2d(C_out, C_out, kernel=3, padding=1, bias=False)
        ├─→ BatchNorm2d(C_out)
        └─→ ReLU(inplace=True)
             │
             ├─── SAVE as skip connection (before pooling)
             │
             └─→ MaxPool2d(kernel=2, stride=2)
```

**Stage-by-stage dimensions:**

```
Input:    (B,   3, 256, 256)
Stage 1:  (B,  64, 256, 256)  → MaxPool → (B,  64, 128, 128)  [skip_1]
Stage 2:  (B, 128, 128, 128)  → MaxPool → (B, 128,  64,  64)  [skip_2]
Stage 3:  (B, 256,  64,  64)  → MaxPool → (B, 256,  32,  32)  [skip_3]
Stage 4:  (B, 512,  32,  32)  → MaxPool → (B, 512,  16,  16)  [skip_4]
```

### 6.3 Bottleneck

```
Input:  (B, 512, 16, 16)
Conv3×3-BN-ReLU × 2
Output: (B, 1024, 16, 16)    ← deepest feature map
```

### 6.4 Absolute Difference at Skip Connections

For each encoder level `i`, both images produce feature maps that are directly subtracted:

```
skip_diff_i = |enc_T1_level_i  −  enc_T2_level_i|
```

**Why absolute value?**  
The sign of `(feat_T1 - feat_T2)` depends on the direction of change (new building vs demolished building). Taking the absolute value makes the signal direction-agnostic — the decoder only needs to know WHERE things changed, not which direction.

**Parameter count:** The absolute difference operation is **parameter-free**. This is a key advantage — U-Net achieves its strong performance without adding any learnable parameters to the comparison step.

### 6.5 Decoder Architecture

Each decoder stage upsamples using `ConvTranspose2d` then concatenates the corresponding skip difference:

```
DECODER STAGE TEMPLATE:
    Input from previous stage (B, C_in, H, W)
        │
        ├─→ ConvTranspose2d(C_in, C_in/2, kernel=2, stride=2)   ← spatial ×2
        │   Output: (B, C_in/2, 2H, 2W)
        │
        ├─→ Concatenate with skip_diff_i                          ← channel doubling
        │   Output: (B, C_in, 2H, 2W)
        │
        ├─→ Conv2d(C_in, C_in/2, kernel=3, padding=1)
        ├─→ BatchNorm2d + ReLU
        ├─→ Conv2d(C_in/2, C_in/2, kernel=3, padding=1)
        └─→ BatchNorm2d + ReLU
             Output: (B, C_in/2, 2H, 2W)
```

**Stage-by-stage decoder dimensions:**

```
Bottleneck diff: (B, 1024, 16, 16)
Decoder Stage 1: ConvT(1024→512) + cat(skip_diff_4:512) → conv → (B,  512, 32, 32)
Decoder Stage 2: ConvT(512→256)  + cat(skip_diff_3:256) → conv → (B,  256, 64, 64)
Decoder Stage 3: ConvT(256→128)  + cat(skip_diff_2:128) → conv → (B,  128, 128, 128)
Decoder Stage 4: ConvT(128→64)   + cat(skip_diff_1:64)  → conv → (B,   64, 256, 256)
Output head:     Conv1×1(64→1)                                 → (B,    1, 256, 256)
```

### 6.6 BatchNorm in the Encoder

```
BN(x) = γ · (x − μ_B) / √(σ²_B + ε)  +  β

where:
  μ_B   = mean over (B, H, W) dimensions, per channel
  σ²_B  = variance over (B, H, W) dimensions, per channel
  γ, β  ∈ ℝ^C   learnable scale and shift, per channel
  ε     = 1×10⁻⁵
```

BatchNorm normalises each feature channel across the batch. It acts as a regulariser and enables faster training by reducing internal covariate shift.

### 6.7 Parameter Breakdown

| Component | Parameters | % |
|---|---|---|
| Encoder (4 stages + bottleneck) | 18,847,168 | 60.7% |
| Decoder (4 stages) | 12,190,465 | 39.3% |
| Diff operation | 0 | 0.0% |
| Output head | 65 | ~0% |
| **Total** | **31,037,633** | |

### 6.8 Detailed Per-Stage Parameters

```
Encoder:
  Stage 1 (3→64):    Conv(3,64,3,1):1,792  + Conv(64,64,3,1):36,928  =  38,720
  Stage 2 (64→128):  Conv(64,128):73,856   + Conv(128,128):147,584   = 221,440
  Stage 3 (128→256): Conv(128,256):295,168 + Conv(256,256):590,080   = 885,248
  Stage 4 (256→512): Conv(256,512):1.18M   + Conv(512,512):2.36M     = 3,539,968
  Bottleneck (512→1024): 2 × Conv blocks                             ≈ 14,156,800

Decoder (includes ConvTranspose + Conv blocks at doubled channels):
  Dec1 (1024→512+skip): ≈ 4.72M  → ≈ 5.24M  total with concat
  Dec2 (512→256+skip):  ≈ 1.18M
  Dec3 (256→128+skip):  ≈ 295K
  Dec4 (128→64+skip):   ≈ 74K
  Output Conv1×1:       65 params
```

---

## 7. Swin Transformer — Complete Architecture

### 7.1 Motivation: Fixing ViT's Quadratic Complexity

Pure ViT computes attention over all N=256 patches: complexity O(N²·d). For larger inputs this becomes prohibitive. Swin Transformer (Liu et al., 2021) restricts attention to local **windows** of size M×M patches, reducing complexity to O(N·M²·d) — linear in image size.

### 7.2 Shifted Window Attention

**Window Multi-head Self-Attention (W-MSA):**  
Divide the feature map into non-overlapping windows of size M×M. Compute self-attention independently within each window. Patches in different windows cannot attend to each other.

**Shifted Window Multi-head Self-Attention (SW-MSA):**  
Shift the window partitioning by (M/2, M/2) pixels. This allows cross-window attention between patches that were in different windows in the W-MSA layer. Alternating W-MSA and SW-MSA across consecutive blocks enables global information flow.

```
Block 2k:   W-MSA  (fixed windows)
Block 2k+1: SW-MSA (shifted windows by M/2 = 3 pixels for M=7)
```

### 7.3 Hierarchical 4-Stage Architecture (Swin-Tiny configuration)

```
Stage 1: PatchEmbed(4×4) + 2 Swin blocks
         In:  (B, 3, 256, 256)
         Out: (B, 64×64, 96)     [1/4 resolution, embed_dim=96, 3 heads]

PatchMerging: 2×2 → 1, channels ×2

Stage 2: 2 Swin blocks
         In:  (B, 32×32, 192)
         Out: (B, 32×32, 192)    [1/8 resolution, 192-dim, 6 heads]

PatchMerging

Stage 3: 6 Swin blocks  ← most computation
         In:  (B, 16×16, 384)
         Out: (B, 16×16, 384)    [1/16 resolution, 384-dim, 12 heads]

PatchMerging

Stage 4: 2 Swin blocks
         In:  (B, 8×8, 768)
         Out: (B, 8×8, 768)      [1/32 resolution, 768-dim, 24 heads]
```

**Depth configuration:** (2, 2, 6, 2) = 12 total blocks, same as ViT-Base.

### 7.4 Scale Adapters

To interface with the same MultiScaleDiffModule used by ViT, all 4 Swin stages are projected to a uniform `(B, 256, 768)` format:

```
Stage 1 (B, 96,  64, 64) → bilinear→(96,16,16)  → Linear(96→768)  → (B, 256, 768)
Stage 2 (B, 192, 32, 32) → bilinear→(192,16,16) → Linear(192→768) → (B, 256, 768)
Stage 3 (B, 384, 16, 16) → identity             → Linear(384→768) → (B, 256, 768)
Stage 4 (B, 768, 8,  8)  → bilinear→(768,16,16) → Linear(768→768) → (B, 256, 768)
```

After scale adaptation, the Swin model feeds into the identical MultiScaleDiffModule and ProgressiveDecoder as the ViT model.

### 7.5 Parameter Breakdown

| Component | Parameters | % |
|---|---|---|
| Swin Backbone (4 stages) | 27,517,818 | 67.8% |
| Scale Adapters | 1,108,992 | 2.7% |
| MultiScaleDiffModule | 10,883,840 | 26.8% |
| ProgressiveDecoder | 1,015,473 | 2.5% |
| **Total** | **40,526,123** | |

### 7.6 Swin vs ViT: Key Differences

| Property | ViT | Swin |
|---|---|---|
| Attention scope | Global (all patches) | Local (M×M window) |
| Complexity | O(N²·d) | O(N·M²·d) |
| Feature hierarchy | Single scale (16×16) | 4 scales (64→8) |
| Inductive bias | None | Locality (window) |
| Patch size | 16×16 | 4×4 (finer) |

---

## 8. Feature Difference Module

### 8.1 FeatureDifferenceModule (per scale)

Given features from both time points at a single scale:
`feat1, feat2` ∈ ℝ^(B × N × 768)

**Strategy: concat_project (4-way comparison)**

```
combined = cat([feat1, feat2, feat1−feat2, feat1⊙feat2], dim=-1)
         ∈ ℝ^(B × N × 3072)      (768 × 4 = 3072)
```

Each component carries different information:
- `feat1`:          "what the scene looked like before (T1)"
- `feat2`:          "what the scene looks like now (T2)"
- `feat1 − feat2`:  directional change signal (positive = present in T1 only, negative = T2 only)
- `feat1 ⊙ feat2`:  element-wise product — high where BOTH agree → highlights stable background

**Compress via MLP:**
```
(B, N, 3072)
  → Linear(3072, 768) + GELU + Dropout(0.1)
  → Linear(768, 256)  + GELU + Dropout(0.1)
  → (B, N, 256)
```

### 8.2 MultiScaleDiffModule (all 4 scales)

```
Input: feats1_list [4 × (B, 256, 768)], feats2_list [4 × (B, 256, 768)]

For each scale i ∈ {0,1,2,3}:
    scale_out_i = FeatureDifferenceModule(feats1[i], feats2[i])
                ∈ ℝ^(B × 256 × 256)

Concatenate along feature dimension:
    combined = cat([scale_out_0, scale_out_1, scale_out_2, scale_out_3], dim=-1)
             ∈ ℝ^(B × 256 × 1024)

Fusion MLP:
    (B, 256, 1024)
      → Linear(1024, 512) + GELU + Dropout(0.1)
      → Linear(512, 256)  + GELU
      → (B, 256, 256)     ← decoder input
```

**Why fuse 4 scales?**
- Scale 0 (shallow): precise edge locations, fine boundary information
- Scale 1 (mid): building part-level features
- Scale 2 (deep): object-level semantics
- Scale 3 (deepest): global context, full scene understanding

The fusion MLP learns how to weight and combine all 4 scales adaptively.

---

## 9. Progressive Decoder

### 9.1 The Upsampling Challenge

After the encoder + diff module we have `diff_feat` ∈ ℝ^(B × 256 × 256):
- 256 tokens, each 256-dimensional
- These tokens correspond to a 16×16 spatial grid (256 = 16×16 patches)
- We need to produce a full `(B, 1, 256, 256)` pixel-level change mask

### 9.2 Architecture

**Step 1 — Reshape tokens to spatial feature map:**
```
(B, 256, 256) → Linear(256→512) + GELU → (B, 256, 512)
             → reshape → (B, 512, 16, 16)    [16×16 spatial grid, 512 channels]
```

**Step 2 — 4 Upsampling Stages (bilinear + Conv):**

```
Stage 1: Upsample(×2, bilinear) → Conv3×3(512→128) + BN + ReLU → Conv3×3(128→128) + BN + ReLU
         (B, 512, 16, 16) → (B, 128, 32, 32)

Stage 2: Upsample(×2, bilinear) → Conv3×3(128→64) + BN + ReLU  → Conv3×3(64→64) + BN + ReLU
         (B, 128, 32, 32) → (B, 64, 64, 64)

Stage 3: Upsample(×2, bilinear) → Conv3×3(64→32) + BN + ReLU   → Conv3×3(32→32) + BN + ReLU
         (B, 64, 64, 64) → (B, 32, 128, 128)

Stage 4: Upsample(×2, bilinear) → Conv3×3(32→16) + BN + ReLU   → Conv3×3(16→16) + BN + ReLU
         (B, 32, 128, 128) → (B, 16, 256, 256)

Output head: Conv1×1(16 → 1) → (B, 1, 256, 256)    ← raw logits
```

### 9.3 Bilinear Upsampling vs Transposed Convolution

**Transposed Convolution** (deconvolution): learnable upsampling, but produces **checkerboard artifacts** due to uneven overlap in the kernel stride pattern.

**Bilinear Upsample + Conv:** Bilinear interpolation handles smooth spatial upsampling; the subsequent learned Conv refines the features. This combination avoids checkerboard artifacts and is the industry-standard approach.

---

## 10. Loss Functions

### 10.1 The Class Imbalance Problem

In LEVIR-CD, changed pixels < 10% of total. With standard Binary Cross-Entropy:

```
Loss(all zeros) = -log(1 - 0.0) × 0.90 + (-log(0.0 + ε)) × 0.10
                ≈ 0 × 0.90 + very_large × 0.10
```

The model quickly learns to minimise loss by predicting all zeros. Result: 90% accuracy, 0% F1. Useless.

### 10.2 Focal Loss

Focal Loss (Lin et al., 2017) was designed specifically for class-imbalanced detection problems.

```
Standard BCE:   L_BCE = -[y·log(p) + (1-y)·log(1-p)]

Focal Loss:     L_FL  = -α_t · (1 - p_t)^γ · log(p_t)

where:
  p     = sigmoid(logit)    (predicted probability for class 1)
  
  p_t   = p        if y = 1    (probability assigned to correct class)
          1 - p    if y = 0

  α_t   = α        if y = 1    (α = 0.25, down-weights dominant class)
          1 - α    if y = 0

  γ     = 2.0      (focusing parameter)
```

**The `(1 - p_t)^γ` modulating factor:**

| Prediction confidence | `p_t` | `(1-p_t)^2` | Effect |
|---|---|---|---|
| Very confident, correct | 0.95 | 0.0025 | Loss reduced 400× |
| Confident, correct | 0.80 | 0.04 | Loss reduced 25× |
| Uncertain | 0.50 | 0.25 | Loss reduced 4× |
| Wrong prediction | 0.10 | 0.81 | Minimal reduction |

This forces the model to focus learning effort on **hard, misclassified examples** rather than the easy, already-correct background pixels.

**pos_weight = 20.0:** Further multiplies the focal BCE contribution for positive (change) pixels by 20, providing an additional strong signal for the minority class.

### 10.3 Dice Loss

```
Dice Coefficient:
  Dice = (2·TP + ε) / (2·TP + FP + FN + ε)    [ε = 1.0, Laplace smoothing]

Dice Loss:
  L_Dice = 1 − Dice

Using soft (continuous) predictions:
  TP_soft = Σ (p_i · y_i)
  FP_soft = Σ (p_i · (1 - y_i))
  FN_soft = Σ ((1 - p_i) · y_i)
```

Dice Loss directly optimises the F1 metric (since Dice = F1 for binary segmentation). It is inherently robust to class imbalance since it considers the ratio of overlap to total positive predictions.

### 10.4 Combined FocalDice Loss (Siamese ViT)

```
L_total = 0.5 · L_Focal + 0.5 · L_Dice
```

The two losses complement each other:
- Focal Loss: pixel-wise, handles hard examples, sensitive to individual misclassifications
- Dice Loss: set-level, optimises the global overlap metric, class-imbalance invariant

### 10.5 BCE-Dice Loss (Siamese U-Net)

```
L_total = L_BCE + L_Dice
```

Standard BCE (without focal modulation) plus Dice Loss. The `pos_weight=20.0` in BCE compensates for class imbalance in the U-Net configuration.

---

## 11. Training Pipeline

### 11.1 Optimizer: AdamW

```
AdamW update rule:

m_t = β_1 · m_{t-1} + (1-β_1) · g_t           [first moment (momentum)]
v_t = β_2 · v_{t-1} + (1-β_2) · g_t²          [second moment (adaptive LR)]

m̂_t = m_t / (1 - β_1^t)                        [bias correction]
v̂_t = v_t / (1 - β_2^t)

θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε) - α · λ · θ_{t-1}
                                                 [last term = weight decay]

β_1 = 0.9, β_2 = 0.999, ε = 1×10⁻⁸
weight_decay λ = 0.05
```

**AdamW vs Adam:** AdamW decouples the weight decay from the gradient update, making it more effective for transformers. In Adam, L2 regularisation is folded into the gradient, scaling incorrectly with the adaptive learning rates. AdamW applies weight decay directly to the parameters, independent of the gradient.

### 11.2 Differential Learning Rates (ViT only)

```
Optimizer parameter groups:
  Group 1 — Encoder (85.8M params):     lr = base_lr × encoder_lr_scale
                                             = 3×10⁻⁴ × 0.5 = 1.5×10⁻⁴

  Group 2 — Diff + Decoder (3.6M params): lr = base_lr
                                               = 3×10⁻⁴
```

**Rationale:** The encoder is very large (86M params) and contributes proportionally large gradient magnitudes. Without scaling, encoder gradients dominate and prevent the smaller diff/decoder components from adapting fast enough. Using half the LR for the encoder balances gradient contributions.

### 11.3 LR Schedule: Cosine Annealing with Linear Warmup

```
Phase 1 — Linear Warmup (epochs 0 to warmup_epochs=20):
    LR(t) = base_lr × (t / warmup_epochs)

Phase 2 — Cosine Decay (epochs warmup_epochs to max_epochs=200):
    LR(t) = min_lr + (base_lr - min_lr) × 0.5 × (1 + cos(π · progress))

    where progress = (t - warmup_epochs) / (max_epochs - warmup_epochs) ∈ [0, 1]
          base_lr  = 3×10⁻⁴
          min_lr   = 1×10⁻⁶
```

**Why warmup?** At epoch 0, random weight initialisation leads to large, inconsistent gradients. Applying full LR immediately causes large, destabilising parameter updates. Warmup gradually increases LR, letting the model reach a stable loss landscape before applying full gradient steps.

**Why cosine over linear decay?** Cosine decay naturally slows down as it approaches the minimum, providing smaller and smaller updates in the final epochs when the model is near convergence. Linear decay would apply the same rate of reduction uniformly.

### 11.4 Gradient Clipping

```
if ‖∇θ‖₂ > max_norm = 1.0:
    ∇θ ← ∇θ × (max_norm / ‖∇θ‖₂)
```

Global gradient norm clipping prevents exploding gradients, especially important during the warmup phase when LR is rising. Applied before each optimizer step.

### 11.5 Mixed Precision Training (FP16)

All forward and backward passes run in FP16 (half precision) using `torch.amp.autocast`. The optimizer states are kept in FP32. This reduces GPU memory by ~50% and speeds up matrix multiplications on Tensor Core hardware (V100 SXM2).

### 11.6 Early Stopping

```
best_f1 = 0.0
no_improve_count = 0
patience = 50   (ViT) / 30 (U-Net)

After each epoch:
  if val_f1 > best_f1:
    best_f1 = val_f1
    save_checkpoint('best_model.pth')
    no_improve_count = 0
  else:
    no_improve_count += 1
    if no_improve_count >= patience:
      stop training
```

---

## 12. Data Augmentation

### 12.1 Full Pipeline (Albumentations, Training)

All spatial transforms are applied **identically** to both images and the mask. Color transforms are applied **only to images** (mask labels are unchanged by lighting conditions).

```
1. PadIfNeeded(min_height=256, min_width=256, border_mode=REFLECT_101)
   → Safety net; no-op for LEVIR's 1024×1024 images

2. RandomCrop(height=256, width=256)
   → Random 256×256 window from 1024×1024 image
   → Applied identically to img1, img2, mask

3. HorizontalFlip(p=0.5)
   → Left-right mirror

4. VerticalFlip(p=0.5)
   → Top-bottom mirror

5. RandomRotate90(p=0.5)
   → Random rotation from {0°, 90°, 180°, 270°}

6. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5)
   → Images only (not mask!)
   → Simulates different seasons, times of day, atmospheric conditions

7. Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
   → Images only (not mask!)
   → ImageNet statistics
```

### 12.2 Validation/Test Pipeline

```
1. CenterCrop(height=256, width=256)
   → Deterministic — always takes the centre region

2. Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

No random transforms at validation/test time — ensures reproducible evaluation.

### 12.3 Multi-Crop Effect

```
Without multi-crop (n_crops=1):
  411 images × 1 crop = 411 patches/epoch
  Batches = 411 / 8 ≈ 51 steps/epoch

With multi-crop (n_crops=4):
  411 images × 4 crops = 1,644 patches/epoch
  Batches = 1,644 / 8 ≈ 190 steps/epoch
```

Each of the 4 crops draws from a different random location in the 1024×1024 image, providing diverse spatial contexts without duplicating any labeled data.

---

## 13. Experimental Results

### 13.1 ViT Ablation Study

| Version | Change Applied | Best F1 | Δ F1 |
|---|---|---|---|
| v0 | Single-scale ViT, BCE loss | 0.2561 | — |
| v1 | FocalDice + pos_weight=20.0 | 0.6655 | +0.409 |
| v2 | Multi-scale feature taps (4 depths) | 0.6458 | −0.020 |
| v3 | Diff LR (×0.5) + patience=50 | 0.7572 | +0.111 |
| **v4** | **n_crops=4, full 1024 random crop** | **0.8236** | **+0.066** |

The temporary drop at v2 occurred because multi-scale features require the differential LR to stabilise encoder training — v2 alone was not enough, but v2+v3 together produced the jump to 0.7572.

### 13.2 Final Comparison: All Three Models

| Metric | Siamese U-Net | Siamese Swin | Siamese ViT |
|---|---|---|---|
| **F1 Score** | **0.8843** | 0.8613 | 0.8236 |
| **IoU** | **0.7927** | 0.7564 | 0.7001 |
| **Precision** | **0.8769** | 0.8671 | 0.8281 |
| **Recall** | **0.8919** | 0.8556 | 0.8191 |
| **Kappa** | **0.8805** | 0.8568 | 0.8176 |
| Best Epoch | 187 | 162 | 195 |
| Parameters | **31.0M** | 40.5M | 89.4M |
| F1 per 10M params | 0.285 | 0.213 | 0.092 |

### 13.3 Training Dynamics

**ViT (200 epochs):**
- Epoch 0: F1 = 0.064 (model predicts all change, Recall=1.0, Precision=0.033)
- Epoch 20: F1 = 0.204 (warmup complete)
- Epoch 100: F1 = 0.757 (active learning phase)
- Epoch 150: F1 = 0.807 (approaching plateau)
- Epoch 195: F1 = 0.8236 (best, early stopping patience not triggered)
- **No overfitting:** val_loss closely tracked train_loss throughout

**Total training time:** ~8 hours 16 minutes on NVIDIA V100 SXM2.

---

## 14. Analysis: Why U-Net Outperforms ViT

### 14.1 Inductive Biases

An **inductive bias** is an assumption built into the model architecture about the structure of the problem. The right inductive bias can dramatically improve sample efficiency.

**U-Net's inductive biases:**

1. **Translation Invariance:** The same Conv filter is applied at every spatial position. A building at (50,50) and a building at (200,200) activate the same filters — the model learns building detection once and it generalises everywhere automatically.

2. **Locality:** Each neuron in a Conv layer sees only a K×K neighbourhood. This matches natural scenes where nearby pixels are more related than distant ones.

3. **Spatial Hierarchy:** MaxPool progressively reduces resolution while doubling channels. Lower layers learn fine features (edges); higher layers learn coarse semantics. This is a perfect match for multi-scale building detection.

4. **Skip Connections Preserve Spatial Precision:** High-frequency edge information flows directly from encoder to decoder at every resolution, enabling precise pixel-level segmentation.

**ViT's lack of inductive biases:**

ViT has NO built-in spatial assumptions. It must learn:
- That nearby patches are related (from the data)
- That building at position A is the same type of object as building at position B
- That rotation and reflection should produce similar outputs

With only 411 training images, ViT does not have enough samples to learn all these spatial invariances from scratch, even with aggressive augmentation.

### 14.2 Dataset Size Mismatch

Original ViT was trained on JFT-300M (300 million images). Our training set has 411 images × 4 crops = 1,644 patches per epoch. This is approximately 182,000× smaller than the intended scale.

CNNs with their built-in spatial priors work well with small datasets. ViT fundamentally requires massive data to overcome its lack of inductive bias.

### 14.3 High-Resolution Segmentation Favours CNNs

Change detection requires precise, pixel-level output masks. The U-Net skip connections directly bridge each encoder resolution to the decoder, providing multi-scale feature access at zero extra cost. ViT's decoder must reconstruct spatial structure from a 16×16 token grid — spatial precision depends entirely on the quality of learned positional encodings.

### 14.4 Parameter Efficiency

```
U-Net:  F1=0.8843, 31.0M params  →  0.02852 F1/million params
Swin:   F1=0.8613, 40.5M params  →  0.02128 F1/million params
ViT:    F1=0.8236, 89.4M params  →  0.00921 F1/million params
```

ViT is 3.1× less parameter-efficient than U-Net for this task and dataset.

### 14.5 When Would ViT Win?

- **Pretrained backbone:** MAE, DINOv2, or SatMAE (pretrained on large satellite image corpora) would give ViT the spatial priors it needs via transfer learning
- **Larger dataset:** 10,000+ training pairs
- **Multi-city generalisation:** ViT's global attention may better handle distribution shift across geographies
- **Classification (not segmentation):** ViT excels at image-level tasks where the CLS token aggregates global information

### 14.6 Why Swin is in the Middle

Swin's window-based attention provides a **partial inductive bias** — locality within windows. Its hierarchical (4-stage) structure mirrors the U-Net encoder's spatial hierarchy. This places it performance-wise between U-Net (full CNN inductive bias) and ViT (no inductive bias), at a size between the two.

---

## 15. Quiz: 30 Questions with Answers

---

### Section A: Change Detection & Dataset (Q1–Q6)

**Q1.** What is the output of a change detection model and why is it called a binary segmentation problem?

> **Answer:** The output is a 2D map (H×W) where each pixel is assigned 0 (no change) or 1 (changed). It is called binary segmentation because every pixel must be classified into exactly one of two classes — changed or unchanged — at the pixel level, not just at the image level.

---

**Q2.** LEVIR-CD has only 411 training image pairs, yet the training step count per epoch is 190. How is this possible, and what technique produces this expansion?

> **Answer:** Multi-crop augmentation (n_crops=4). Each image stem is virtually sampled 4 times per epoch, drawing a different random 256×256 crop each time from the 1024×1024 original. 411 × 4 = 1,644 patches ÷ batch_size=8 = ~190 steps per epoch.

---

**Q3.** A model achieves 92% pixel accuracy on LEVIR-CD but has F1 = 0.0. Explain how this is possible.

> **Answer:** Class imbalance. If changed pixels are only 8% of the dataset, a model predicting ALL zeros (no change anywhere) achieves 92% accuracy (correctly classifying all background pixels) but detects zero change pixels. Since TP=0, F1 = 2·0/(2·0+FP+FN) = 0. Accuracy is an unreliable metric for imbalanced segmentation problems.

---

**Q4.** What does Cohen's Kappa measure and why is it preferred over accuracy for change detection?

> **Answer:** Kappa measures agreement between predictions and ground truth beyond what would be expected by chance: κ = (p_observed − p_chance) / (1 − p_chance). It accounts for the fact that a random predictor on imbalanced data would still achieve high accuracy purely by predicting the majority class. Kappa > 0.8 indicates very strong agreement and is independent of class distribution.

---

**Q5.** Describe the absolute difference operation at U-Net skip connections and explain why absolute value is used rather than signed difference.

> **Answer:** For each encoder level i: `skip_diff_i = |feat_T1_i − feat_T2_i|`. The absolute value is used because the sign of the difference depends on the direction of change — a demolished building produces negative differences in channels where the building was bright, while a new construction produces positive differences. The decoder cares about WHERE things changed, not which direction. |·| collapses both directions into a single non-negative change signal.

---

**Q6.** Why is the validation set used during training, and why are test set results not reported for the LEVIR-CD experiments in this project?

> **Answer:** The validation set is used for early stopping (monitoring val-F1 to detect overfitting) and hyperparameter selection (e.g., threshold, patience). Since hyperparameters were selected based on val performance, the test set must be kept completely unseen to provide an unbiased final estimate. Reporting test results after optimising on val would constitute data leakage — test set evaluation was deferred for that reason.

---

### Section B: ViT Architecture (Q7–Q14)

**Q7.** An input image is 256×256 pixels. ViT uses patch_size=16. Calculate: (a) number of patches, (b) dimension of each flattened patch, (c) sequence length after adding CLS token.

> **Answer:**  
> (a) (256/16) × (256/16) = 16 × 16 = **256 patches**  
> (b) 16 × 16 × 3 = **768 values**  
> (c) 256 + 1 (CLS) = **257 tokens**

---

**Q8.** Why are positional embeddings necessary in ViT? What property of the transformer makes them essential?

> **Answer:** Transformers are **permutation-invariant** — the self-attention mechanism computes pairwise scores between all tokens without regard to their ordering. Without positional embeddings, the model cannot distinguish the patch at position (0,0) from the patch at position (15,15). Positional embeddings (added to patch embeddings) assign each token a unique spatial identity, breaking the permutation symmetry and allowing the model to learn spatially-structured representations.

---

**Q9.** Write the full scaled dot-product attention formula and explain the purpose of the scaling factor √d_k.

> **Answer:**  
> `Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V`  
> The scaling factor √d_k (= √64 = 8 for ViT-Base) is needed because as d_k grows, the dot products Q·K^T grow in variance proportional to d_k, pushing the softmax into saturation regions where gradients are nearly zero. Dividing by √d_k normalises the variance back to approximately 1.0, maintaining healthy gradient flow.

---

**Q10.** ViT-Base uses h=12 attention heads, each with d_k=64. Why use 12 heads instead of one head with d_k=768?

> **Answer:** Multiple heads allow the model to attend to different relationship types simultaneously. Head 1 might learn spatial proximity, head 6 might learn semantic similarity, head 11 might learn texture correlation. Each head operates in a 64-dimensional subspace, learning a different projection of Q, K, V. With a single 768-dim head, the model is forced to use one type of attention. The final output projection W_O combines all 12 views into a unified representation.

---

**Q11.** What is the Pre-LayerNorm design and why does it produce more stable training than Post-LayerNorm for deep networks?

> **Answer:**  
> Post-LN: `x' = LayerNorm(x + MHSA(x))` — gradients must flow through LayerNorm before reaching the residual connection, causing gradient instability at depth.  
> Pre-LN: `x' = x + MHSA(LayerNorm(x))` — the residual stream `x` flows through the addition unimpeded; gradients propagate cleanly through the shortcut. This enables stable training of 12 deep blocks from random initialisation without gradient vanishing or explosion.

---

**Q12.** ViT uses 12 transformer blocks. How many parameters does the MLP sub-block contribute per block and in total across all 12 blocks?

> **Answer:**  
> fc1: 768 × 3072 + 3072 = 2,362,368  
> fc2: 3072 × 768 + 768 = 2,362,368  
> Per block: **4,724,736**  
> 12 blocks total: **56,696,832**

---

**Q13.** Explain multi-scale feature tapping in ViT. At which block indices are features extracted and what does each scale capture?

> **Answer:** Features are tapped at blocks indexed by [depth//4−1, depth//2−1, 3·depth//4−1, depth−1] = [2, 5, 8, 11] for depth=12.  
> - Block 2 (Scale 0): local texture, edges, patch boundary information  
> - Block 5 (Scale 1): structural parts, shapes, local colour patterns  
> - Block 8 (Scale 2): semantic building components, spatial context  
> - Block 11 (Scale 3): object-level semantics, full global context  
> Each tap has a dedicated LayerNorm except scale 3 which reuses the final model norm.

---

**Q14.** The ViT encoder has 85,844,736 parameters but the diff module has only 2,556,928. Why is the ratio 96.2% / 2.9% — and is this a problem?

> **Answer:** The 12 transformer blocks each have ~7M parameters (MHSA + MLP), adding up to 85M for the encoder. The diff module only needs to combine and project features — a much simpler operation requiring far fewer parameters. This ratio is not inherently a problem; the encoder's job (learning rich visual representations from raw pixels) is fundamentally harder than the diff module's job (computing a change signal from already-extracted features). The decoder similarly only needs 1% of parameters to upsample a 16×16 token grid to 256×256 pixels.

---

### Section C: U-Net Architecture (Q15–Q19)

**Q15.** Trace the spatial dimensions of a 256×256 input through the complete U-Net encoder (all 4 stages + bottleneck). State the shape at each stage output and at the bottleneck.

> **Answer:**  
> Input: (B, 3, 256, 256)  
> After Stage 1 + MaxPool: (B, 64, 128, 128)  
> After Stage 2 + MaxPool: (B, 128, 64, 64)  
> After Stage 3 + MaxPool: (B, 256, 32, 32)  
> After Stage 4 + MaxPool: (B, 512, 16, 16)  
> Bottleneck output: (B, 1024, 16, 16)

---

**Q16.** Why does the U-Net decoder concatenate skip differences rather than just upsampling from the bottleneck alone?

> **Answer:** The encoder's MaxPool operations discard spatial information — fine details like exact building edges are progressively lost as we go deeper. The bottleneck captures high-level semantics but lacks spatial precision. Skip connections provide fine-grained spatial detail from each encoder resolution directly to the corresponding decoder stage. For change detection specifically, skip diffs carry the change signal at every scale — the decoder receives change information at both 256×256 (fine) and 32×32 (coarse) resolutions simultaneously, enabling precise boundary delineation.

---

**Q17.** U-Net has 31M parameters and achieves F1=0.8843. ViT has 89M parameters and achieves F1=0.8236. Calculate the F1 per million parameters for each model. What does this tell us?

> **Answer:**  
> U-Net: 0.8843 / 31.0 = **0.02853** F1 per million params  
> ViT: 0.8236 / 89.4 = **0.00921** F1 per million params  
> U-Net is **3.1×** more parameter-efficient. This demonstrates that a well-matched architecture with the right inductive biases (translation invariance, local hierarchy, skip connections) outperforms a much larger model that lacks those biases when trained on small datasets.

---

**Q18.** What is BatchNorm and how does it differ from LayerNorm? Which is used in U-Net and which in ViT?

> **Answer:**  
> **BatchNorm:** normalises over (B, H, W) per channel — statistics computed across the batch. Requires large batch sizes; behaviour differs between train (running stats) and eval (moving average stats). Used in **U-Net** after each Conv layer.  
> **LayerNorm:** normalises over the feature dimension (C or d) per token per sample — statistics computed per sample. Batch-size independent; same behaviour at train/eval. Standard for transformers. Used in **ViT** before each MHSA and MLP sub-layer.

---

**Q19.** Why is the difference operation in U-Net parameter-free, and what are the implications of this for the total parameter count?

> **Answer:** Absolute difference `|feat1 − feat2|` is a fixed mathematical operation with no learnable parameters — it requires no weights, biases, or normalisations. This means U-Net achieves its change detection capability with only 31M parameters (encoder + decoder), without any overhead for a separate learned comparison module. The ViT model requires an additional 2.56M parameters in its MultiScaleDiffModule to learn the same type of comparison. U-Net's implicit comparison is not only parameter-free but also architecturally elegant.

---

### Section D: Loss Functions & Training (Q20–Q25)

**Q20.** Write the Focal Loss formula and explain each term. For a correct easy prediction (p_t=0.95) vs a hard wrong prediction (p_t=0.05), calculate the modulating factor (1−p_t)^2 for each.

> **Answer:**  
> `L_FL = −α_t · (1 − p_t)^γ · log(p_t)`  
> - α_t: class weighting (down-weights majority class predictions)  
> - (1−p_t)^γ: modulating factor — reduces loss for confident correct predictions  
> - log(p_t): standard cross-entropy term  
>   
> Easy prediction (p_t=0.95): (1−0.95)^2 = (0.05)^2 = **0.0025** (loss reduced 400×)  
> Hard prediction (p_t=0.05): (1−0.05)^2 = (0.95)^2 = **0.9025** (minimal reduction)  
>   
> This shows focal loss concentrates the training signal on hard examples.

---

**Q21.** Calculate the Dice coefficient for a prediction that has: TP=500, FP=100, FN=50. Use ε=1.

> **Answer:**  
> Dice = (2·TP + ε) / (2·TP + FP + FN + ε)  
> = (2·500 + 1) / (2·500 + 100 + 50 + 1)  
> = 1001 / 1151  
> = **0.8697**  
> Dice Loss = 1 − 0.8697 = **0.1303**

---

**Q22.** Explain linear warmup in the LR schedule. Why is training unstable without it when starting from random initialisation?

> **Answer:** Linear warmup linearly increases LR from near-zero to base_lr over the first `warmup_epochs` epochs. Without warmup: random initialisation produces large, inconsistent gradients (the loss landscape is steep and noisy at initialisation). Applying full LR=3×10⁻⁴ immediately causes large parameter updates that can push the model into poor local minima or cause divergence (NaN loss). Warmup starts with very small LR (≈1.5×10⁻⁵), allowing the model to make small, stable initial adjustments until it reaches a region where the loss landscape is smoother, after which full LR can be safely applied.

---

**Q23.** What is gradient clipping and at what threshold is it applied in this project? When does it become critical?

> **Answer:** Gradient clipping rescales the gradient vector when its L2 norm exceeds a threshold: if ‖∇θ‖₂ > 1.0, then ∇θ ← ∇θ × (1.0 / ‖∇θ‖₂). This prevents exploding gradients — a common failure mode in deep transformer training. It is most critical during the warmup phase (epochs 0–20) when LR is rising and gradients are largest, and at any point where a hard batch produces a very large loss.

---

**Q24.** What is differential learning rate and why is it only applied to the ViT model and not the U-Net?

> **Answer:** Differential LR assigns different learning rates to different parameter groups. For ViT: encoder (86M params) gets LR×0.5=1.5×10⁻⁴; diff+decoder (3.6M params) get full LR=3×10⁻⁴. This is needed for ViT because the encoder's large gradient norms (proportional to its parameter count) would dominate training without scaling. U-Net does not need differential LR because its encoder (18.8M) and decoder (12.2M) are more balanced in size and both use the same simple CNN operations with similar gradient magnitudes — no one component dominates.

---

**Q25.** The FocalDice loss combines two losses with equal weight (0.5 each). Conceptually, why are these two losses complementary rather than redundant?

> **Answer:** They optimise different properties:  
> - **Focal Loss** is pixel-wise — it penalises individual misclassified pixels, with extra weight on hard examples. It is sensitive to individual errors but does not directly optimise the global overlap metric (F1).  
> - **Dice Loss** is set-level — it measures the ratio of intersection to union, directly optimising the F1/Dice metric. It is inherently imbalance-invariant but may be less sensitive to boundary precision.  
> Together: Focal handles hard individual pixels; Dice ensures the global overlap is maximised. Focal can drive recall on rare change pixels; Dice prevents precision collapse by penalising excessive false positives.

---

### Section E: Architecture Comparison & Theory (Q26–Q30)

**Q26.** What is an inductive bias in machine learning? Give one example each for U-Net and ViT.

> **Answer:** An inductive bias is an assumption built into the model architecture about the structure of the problem. It allows the model to generalise from limited data by constraining the hypothesis space.  
> - **U-Net inductive bias:** Translation invariance via shared Conv filters — the same building detector works at any spatial location automatically.  
> - **ViT inductive bias (lack of):** ViT has essentially no spatial inductive bias — every patch attends to every other equally regardless of spatial distance. It must learn spatial structure purely from data, requiring much larger datasets to generalise.

---

**Q27.** Swin Transformer uses window-based attention of size 7×7. What is the computational complexity of W-MSA vs full self-attention (ViT), and why does this matter for high-resolution inputs?

> **Answer:**  
> Full self-attention (ViT): O(N²·d) where N = total patches  
> Window attention (Swin): O(N · M² · d/N · d) ≈ O(N · M²) where M = window size  
> For N=256 patches, M=7: Full = O(256²) = O(65,536); Window = O(256 × 49) ≈ O(12,544) — roughly 5× less.  
> At higher resolution (e.g., 512×512 → N=1024 patches), this gap becomes 4× worse for full attention (O(1024²)=O(1,048,576) vs O(1024×49)=O(50,176)) — 21× difference. Window attention scales linearly with image size, enabling application to high-resolution satellite imagery.

---

**Q28.** The ProgressiveDecoder uses bilinear upsampling + Conv instead of transposed convolution. Why?

> **Answer:** Transposed convolution (deconvolution) produces **checkerboard artifacts** — grid-like patterns caused by uneven overlap in the stride pattern. Where kernel entries overlap 4 times vs 1 time, the output has systematic high/low amplitude variations that are visible as a grid. Bilinear upsampling + Conv avoids this: bilinear interpolation produces smooth spatial upsampling based on weighted averages of neighbours, and the subsequent learned Conv(3×3) refines features at the new resolution. This combination is now the industry standard for segmentation decoders.

---

**Q29.** The Siamese ViT uses the 4-way concat_project difference: [feat1, feat2, feat1−feat2, feat1⊙feat2]. Why is feat1⊙feat2 (element-wise product) included? What would be lost without it?

> **Answer:** The element-wise product feat1⊙feat2 is high when both feature vectors agree (both T1 and T2 have similar activation patterns at that position) — it highlights stable, unchanged regions. This effectively lets the model learn "these pixels look the same in both images → not changed." Without the product term, the module only has access to feat1, feat2, and feat1−feat2. The subtraction captures what changed, but can be ambiguous when features are large in magnitude for unrelated reasons. The product provides a complementary signal: low product = high disagreement = likely changed. Together the four terms give the most expressive comparison possible without increasing depth.

---

**Q30.** Summarise the key findings of this project in terms of three lessons a researcher should take away about architecture selection for satellite change detection.

> **Answer:**  
> **Lesson 1 — Match architecture to task constraints:** For small-dataset, high-resolution binary segmentation, CNN architectures (U-Net) with built-in spatial inductive biases outperform large transformer models trained from scratch. U-Net (31M params) achieved F1=0.8843 vs ViT (89M params) at F1=0.8236.  
>   
> **Lesson 2 — Loss function design is as important as architecture:** The single largest F1 improvement (+0.41) came from switching BCE to FocalDice loss — not from increasing model capacity. Addressing class imbalance at the loss level is critical for change detection.  
>   
> **Lesson 3 — Free data is often the best data:** Multi-crop augmentation (n_crops=4) added +0.066 F1 (+8.7% relative gain) at zero annotation cost. Architectural improvements and augmentation strategies are complementary — both should be explored before increasing model size.

---

*End of Research Booklet*

---

**Project Repository:** `github.com/parvpatodia/vit-from-scratch`  
**Cluster:** Northeastern University Explorer HPC (NVIDIA V100 SXM2)  
**Best Result:** Siamese U-Net — F1 = 0.8843 on LEVIR-CD validation set
