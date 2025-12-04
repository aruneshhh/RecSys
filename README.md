# 📦 Recommendation System — First Iteration (Embedding + MLP Hybrid Recommender)

*A PyTorch-based implicit-feedback recommendation system (first working prototype)*

This project implements the **first complete working version** of a neural recommendation system using **PyTorch**, **custom datasets**, and a **hybrid collaborative-filtering architecture**.
The aim of this iteration is to produce a clean, reproducible, end-to-end pipeline that loads user–item interactions → builds index mappings → trains a hybrid neural model → evaluates performance on validation data.

This version forms the **baseline recommender system** that your team can iterate on with more advanced architectures (attention, transformers, sequence modeling, etc.).

---

# 🚀 Project Overview

This repository contains:

* A single Python script: **`RecSys.py`**
* A **custom PyTorch Dataset** for user–item interactions
* An **embedding-based recommender model** combining:

  * Dot-product similarity
  * A Multi-Layer Perceptron (Deep Component)
* A full **training loop**
* A **validation loop** with accuracy metrics
* Automatic **ID encoding** + preprocessing

This prototype establishes a clear and modular foundation for future improvements.

---

# 📁 File Structure

```
RecSys.py
train.csv
val.csv
```

> Your script expects interaction CSV files with at least:
> `userId, itemId, target`

---

# 🔧 Main Components & Explanation (Step-by-Step Summary)

Below is a structured explanation of what **your script actually does**, following the same style as the example README.

---

# 1️⃣ Imports & Initial Setup

The script imports:

* `pandas`, `numpy` — tabular data handling
* `torch`, `nn`, `DataLoader` — model training + batching
* `tqdm` — training progress bar
* `os` — file paths
* `math` — utilities

It automatically detects **CPU/GPU** and sets the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This allows training without any manual hardware configuration.

---

# 2️⃣ Loading the Data

The script reads:

```python
train.csv
val.csv
```

Each CSV requires the columns:

| Column | Description                           |
| ------ | ------------------------------------- |
| userId | Raw user ID from the dataset          |
| itemId | Raw item ID                           |
| target | Implicit label: 0/1 or click/no-click |

The system **does not** rely on timestamp order — this version implements a **pointwise implicit recommender**, not a sequential one.

---

# 3️⃣ User/Item Encoding

To use embedding layers, the model needs **dense, continuous indices**.

The script scans the dataset to build two mappings:

```
raw_user_id → internal_user_index
raw_item_id → internal_item_index
```

Why?

* Embeddings require IDs starting at 0
* Memory remains compact
* Training is faster and stable

The mappings are stored inside the dataset class and used throughout training.

---

# 4️⃣ Custom PyTorch Dataset

The core dataset class is:

```python
class RecSysDataset(torch.utils.data.Dataset)
```

It returns a tuple for each training sample:

```
(user_idx, item_idx, target)
```

All values are converted to PyTorch tensors.

This forms the backbone of the training pipeline, ensuring:

* Efficient batching
* GPU-optimized data loading
* Clean separation of data and model logic

---

# 5️⃣ DataLoader Setup

Next, the script constructs:

```python
train_loader = DataLoader(...)
val_loader   = DataLoader(...)
```

With batching, shuffling (for training), and correct tensor formatting.

This enables:

* Mini-batch gradient descent
* Batched predictions
* Faster GPU utilization

---

# 6️⃣ Recommender Model Architecture (Your Hybrid Design)

The script defines a hybrid recommender:

```python
class HybridRecSys(nn.Module)
```

It combines two core ideas:

## ✔ A. Embedding-Based Collaborative Filtering

Two embedding layers:

```python
user_embedding = Embedding(num_users, embed_dim)
item_embedding = Embedding(num_items, embed_dim)
```

Purpose:

* Learn latent user preferences
* Learn latent item attributes
* Capture collaborative patterns

---

## ✔ B. A Deep MLP Interaction Component

After embeddings are extracted, they are concatenated:

```
[ user_embed || item_embed ]
```

Then passed through an MLP with:

* Linear layers
* ReLU activation
* Optional dropout

This component learns nonlinear patterns such as:

* “Users who like A and B also like C only when item metadata is similar”
* Complex interaction relationships not captured by dot-product alone

---

## ✔ C. Dot Product Similarity Component

Parallel to the MLP, the script computes:

```
score = dot(user_embedding, item_embedding)
```

This is the classical Matrix Factorization relevance measure.

It provides:

* Stable gradients
* Good cold-start generalization
* Strong baseline performance

---

## ✔ D. Final Output = Deep Score + Dot Score

The final predicted logit is:

```
output = mlp_score + dot_score
```

This hybrid scoring mechanism is used in many production recommenders (YouTube, Amazon, TikTok).

---

# 7️⃣ Training Pipeline

The training loop performs:

1. Forward pass
2. Compute **BCEWithLogitsLoss** (more stable than BCE)
3. Backpropagation
4. Update weights with Adam optimizer
5. Track moving averages of:

   * Loss
   * Accuracy
   * Throughput (samples/sec)

The script prints clean training logs per epoch.

---

# 8️⃣ Validation Pipeline

After each epoch:

* Model is put in `eval()` mode
* Gradients disabled (`torch.no_grad()`)
* Validation loss
* Validation accuracy

This allows detection of:

* Overfitting
* Underfitting
* Training inconsistencies

---

# 🧠 Model Summary

| Component      | Description                      |
| -------------- | -------------------------------- |
| User Embedding | `num_users × embed_dim`          |
| Item Embedding | `num_items × embed_dim`          |
| MLP Layers     | Nonlinear user–item interactions |
| Dot Product    | Classic collaborative filtering  |
| Loss Function  | BCEWithLogitsLoss                |
| Optimizer      | Adam                             |
| Hardware       | GPU if available                 |

---

# ▶️ How to Run the Project

### 1. Install dependencies:

```
pip install torch pandas numpy tqdm
```

### 2. Prepare the CSV files:

They must include:

```
userId, itemId, target
```

### 3. Run the script:

```
python RecSys.py
```

