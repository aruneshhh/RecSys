import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import pickle
from typing import Dict
from collections import Counter, defaultdict
import math

# -------------------------
# CONFIG
# -------------------------
TRAIN_PATH = "train.csv"
VAL_PATH   = "val.csv"
TEST_PATH  = "test.csv"

RAW_PATH = "/Users/aruneshvenkatesan/Documents/RU_DATA/IR/RecSys/raw_review_All_Beauty/All_Beauty.jsonl"
META_PATH = "/Users/aruneshvenkatesan/Documents/RU_DATA/IR/RecSys/raw_review_All_Beauty/meta_All_Beauty.jsonl"

EMBED_DIM = 64
BATCH_SIZE = 256
NUM_EPOCHS = 50
NUM_NEGATIVES = 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ITEM_TEXT = True
ITEM_TEXT_DIM = 128

# -------------------------
# STEP 0: BUILD TRAIN/VAL/TEST FROM RAW
# -------------------------
if not (os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH) and os.path.exists(TEST_PATH)):
    print("Train/val/test CSVs not found. Building from raw JSONL...")

    raw_df = pd.read_json(RAW_PATH, lines=True)
    df = raw_df[["user_id", "parent_asin", "rating", "timestamp"]].dropna()

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    splits = []
    frac_train = 0.8
    frac_val = 0.1

    for uid, g in df.groupby("user_id"):
        n = len(g)
        if n < 3:
            user_splits = ["train"] * n
        else:
            n_train = max(1, int(n * frac_train))
            n_val = max(1, int(n * frac_val))
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1
            n_test = n - n_train - n_val
            user_splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        splits.extend(user_splits)

    df["split"] = splits

    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df   = df[df["split"] == "val"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])

    print("Split sizes:",
          "train =", len(train_df),
          "val =", len(val_df),
          "test =", len(test_df))

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
else:
    print("Found existing train/val/test CSVs. Skipping raw split.")


# -------------------------
# UTILS: METRICS
# -------------------------
def recall_at_k(true_items, ranked_items, k):
    topk = set(ranked_items[:k])
    hits = len(topk & true_items)
    return hits / max(1, len(true_items))

def hitrate_at_k(true_items, ranked_items, k):
    topk = set(ranked_items[:k])
    return 1.0 if len(topk & true_items) > 0 else 0.0

def ndcg_at_k(true_items, ranked_items, k):
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in true_items:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(true_items), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg

# -------------------------
# DATA LOADING
# -------------------------
print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Ensure label exists (implicit feedback: rating >= 4 -> positive)
if "label" not in train_df.columns:
    train_df["label"] = (train_df["rating"] >= 4).astype(int)
    val_df["label"]   = (val_df["rating"]   >= 4).astype(int)
    test_df["label"]  = (test_df["rating"]  >= 4).astype(int)

# Filter val/test to users/items seen in train
train_users = set(train_df["user_id"])
train_items = set(train_df["parent_asin"])

val_df = val_df[
    val_df["user_id"].isin(train_users)
    & val_df["parent_asin"].isin(train_items)
].reset_index(drop=True)

test_df = test_df[
    test_df["user_id"].isin(train_users)
    & test_df["parent_asin"].isin(train_items)
].reset_index(drop=True)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# -------------------------
# ID MAPPINGS (from TRAIN only)
# -------------------------
print("Building ID mappings from train...")
unique_train_users = train_df["user_id"].unique()
unique_train_items = train_df["parent_asin"].unique()

user2id: Dict[str, int] = {u: i for i, u in enumerate(unique_train_users)}
item2id: Dict[str, int] = {it: i for i, it in enumerate(unique_train_items)}

num_users = len(user2id)
num_items = len(item2id)

print("Num users:", num_users)
print("Num items:", num_items)

# Save mappings
with open("user2id.pkl", "wb") as f:
    pickle.dump(user2id, f)
with open("item2id.pkl", "wb") as f:
    pickle.dump(item2id, f)

# Inverse mapping for evaluation / bucketing
id2user = {v: k for k, v in user2id.items()}
id2item = {v: k for k, v in item2id.items()}

# -------------------------
# SPARSITY STATS & BUCKETS
# -------------------------
user_inter_count = train_df.groupby("user_id")["parent_asin"].nunique()
item_inter_count = train_df.groupby("parent_asin")["user_id"].nunique()

def user_bucket(u):
    c = user_inter_count.get(u, 0)
    if c <= 2:
        return "cold"
    elif c <= 10:
        return "medium"
    else:
        return "hot"

def item_bucket(i):
    c = item_inter_count.get(i, 0)
    if c <= 2:
        return "tail"
    elif c <= 20:
        return "mid"
    else:
        return "head"

print("User interaction stats:")
print(user_inter_count.describe())
print("Item interaction stats:")
print(item_inter_count.describe())

# -------------------------
# TORCH DATASET
# -------------------------
class RecSysDataset(Dataset):
    """
    Dataset for implicit-feedback training with sampled negatives.
    Uses only positive interactions (label == 1) from df.
    """
    def __init__(self, df, user2id, item2id, num_negatives=3):
        # Keep only positives
        self.df = df[df["label"] == 1].reset_index(drop=True)
        self.user2id = user2id
        self.item2id = item2id
        self.num_negatives = num_negatives

        # Precompute list of item indices for negative sampling
        self.all_items = list(item2id.values())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u = self.user2id[row["user_id"]]
        i_pos = self.item2id[row["parent_asin"]]

        # Sample negatives
        neg_items = []
        while len(neg_items) < self.num_negatives:
            j = random.choice(self.all_items)
            if j != i_pos:
                neg_items.append(j)

        return {
            "user": torch.tensor(u, dtype=torch.long),
            "pos_item": torch.tensor(i_pos, dtype=torch.long),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
        }

# -------------------------
# MODELS
# -------------------------
class TwoTowerIDOnly(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def get_user_embedding(self, user_ids):
        return self.user_emb(user_ids)

    def get_item_embedding(self, item_ids, item_text_feats=None):
        # item_text_feats ignored for ID-only model
        return self.item_emb(item_ids)

    def forward(self, user_ids, item_ids, item_text_feats=None):
        u = self.get_user_embedding(user_ids)
        i = self.get_item_embedding(item_ids, item_text_feats)
        scores = (u * i).sum(dim=1)
        return torch.sigmoid(scores)

class TwoTowerWithItemText(nn.Module):
    """
    User tower: ID embedding only.
    Item tower: ID embedding + projected text embedding (concatenated).
    """
    def __init__(self, num_users, num_items,
                 user_emb_dim=64,
                 item_id_emb_dim=32,
                 item_text_dim=128,
                 out_dim=64):
        super().__init__()
        assert item_id_emb_dim < out_dim, "item_id_emb_dim must be < out_dim"

        self.user_emb = nn.Embedding(num_users, out_dim)

        self.item_id_emb = nn.Embedding(num_items, item_id_emb_dim)
        self.item_text_proj = nn.Linear(item_text_dim, out_dim - item_id_emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)

    def get_user_embedding(self, user_ids):
        return self.user_emb(user_ids)

    def get_item_embedding(self, item_ids, item_text_feats=None):
        id_vec = self.item_id_emb(item_ids)
        if item_text_feats is None:
            text_vec = torch.zeros(
                (id_vec.size(0), self.item_text_proj.out_features),
                device=id_vec.device
            )
        else:
            text_vec = self.item_text_proj(item_text_feats)
        return torch.cat([id_vec, text_vec], dim=1)

    def forward(self, user_ids, item_ids, item_text_feats=None):
        u = self.get_user_embedding(user_ids)
        i = self.get_item_embedding(item_ids, item_text_feats)
        scores = (u * i).sum(dim=1)
        return torch.sigmoid(scores)

# -------------------------
# BUILD ITEM TEXT FEATURES FROM META (RUN ONCE)
# -------------------------
def build_item_text_features_from_meta(meta_path, item2id,
                                       text_dim=ITEM_TEXT_DIM,
                                       save_path="item_text_features.npy"):
    """
    Build TF-IDF + SVD item text embeddings from meta_All_Beauty.jsonl.
    Uses fields: title, features, description, categories, details, store.
    Requires scikit-learn installed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print(f"Loading metadata from {meta_path} ...")
    meta_df = pd.read_json(meta_path, lines=True)

    # Helper: convert list/dict/NaN to a flat string
    def to_str(x):
        if isinstance(x, list):
            return " ".join(str(t) for t in x)
        if isinstance(x, dict):
            return " ".join(f"{k} {v}" for k, v in x.items())
        if pd.isna(x):
            return ""
        return str(x)

    # Build a big text field per row using multiple metadata fields
    title_str    = meta_df["title"].fillna("") if "title" in meta_df.columns else ""
    features_str = meta_df["features"].apply(to_str) if "features" in meta_df.columns else ""
    desc_str     = meta_df["description"].apply(to_str) if "description" in meta_df.columns else ""
    cats_str     = meta_df["categories"].apply(to_str) if "categories" in meta_df.columns else ""
    details_str  = meta_df["details"].apply(to_str) if "details" in meta_df.columns else ""
    store_str    = meta_df["store"].fillna("") if "store" in meta_df.columns else ""

    meta_df["text_all"] = (
        title_str.astype(str) + " " +
        features_str.astype(str) + " " +
        desc_str.astype(str) + " " +
        cats_str.astype(str) + " " +
        details_str.astype(str) + " " +
        store_str.astype(str)
    ).str.strip()

    # Ensure we have parent_asin to join with item2id
    if "parent_asin" not in meta_df.columns:
        raise ValueError("Metadata file is missing 'parent_asin' field.")

    # One row per parent_asin
    item_text = (
        meta_df[["parent_asin", "text_all"]]
        .dropna(subset=["parent_asin"])
        .drop_duplicates(subset=["parent_asin"])
        .reset_index(drop=True)
    )

    corpus = item_text["text_all"].tolist()
    print(f"Building TF-IDF on {len(corpus)} items...")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf = vectorizer.fit_transform(corpus)
    print("TF-IDF matrix shape:", tfidf.shape)

    svd = TruncatedSVD(n_components=text_dim, random_state=42)
    item_vecs = svd.fit_transform(tfidf)
    print("SVD-reduced matrix shape:", item_vecs.shape)

    num_items = len(item2id)
    item_features = np.zeros((num_items, text_dim), dtype=np.float32)

    asin_to_row = {asin: idx for idx, asin in enumerate(item_text["parent_asin"])}

    missing = 0
    for asin, idx in item2id.items():
        if asin in asin_to_row:
            item_features[idx] = item_vecs[asin_to_row[asin]]
        else:
            missing += 1
    print(f"Items without meta {missing} / {num_items}")

    np.save(save_path, item_features)
    print("Saved", save_path, "with shape", item_features.shape)

# -------------------------
# TRAINING SETUP
# -------------------------
train_dataset = RecSysDataset(train_df, user2id, item2id, num_negatives=NUM_NEGATIVES)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# Load or build item text features from metadata
item_text_tensor = None
if USE_ITEM_TEXT:
    if not os.path.exists("item_text_features.npy"):
        print("item_text_features.npy not found. Building from metadata ...")
        build_item_text_features_from_meta(
            META_PATH,
            item2id,
            text_dim=ITEM_TEXT_DIM,
            save_path="item_text_features.npy"
        )
    item_text_features = np.load("item_text_features.npy")
    assert item_text_features.shape == (num_items, ITEM_TEXT_DIM)
    item_text_tensor = torch.tensor(item_text_features, dtype=torch.float32, device=DEVICE)

# Choose model
if USE_ITEM_TEXT:
    model = TwoTowerWithItemText(
        num_users=num_users,
        num_items=num_items,
        user_emb_dim=EMBED_DIM,
        item_id_emb_dim=32,
        item_text_dim=ITEM_TEXT_DIM,
        out_dim=EMBED_DIM
    ).to(DEVICE)
else:
    model = TwoTowerIDOnly(num_users, num_items, emb_dim=EMBED_DIM).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()

# -------------------------
# TRAINING LOOP
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, item_text_tensor=None):
    model.train()
    total_loss = 0.0
    total_pos_loss = 0.0
    total_neg_loss = 0.0

    for batch in tqdm(loader, desc="Train"):
        user = batch["user"].to(device)
        pos_item = batch["pos_item"].to(device)
        neg_items = batch["neg_items"].to(device)  # (B, K)

        optimizer.zero_grad()

        # Prepare text features if available
        pos_text = item_text_tensor[pos_item] if item_text_tensor is not None else None

        B, K = neg_items.shape
        neg_flat = neg_items.view(-1)
        neg_text = item_text_tensor[neg_flat] if item_text_tensor is not None else None

        # Positive scores
        pos_scores = model(user, pos_item, pos_text)
        pos_labels = torch.ones_like(pos_scores)
        pos_loss = criterion(pos_scores, pos_labels)

        # Negative scores (vectorized)
        user_rep = user.unsqueeze(1).expand(-1, K).reshape(-1)
        neg_scores = model(user_rep, neg_flat, neg_text)
        neg_labels = torch.zeros_like(neg_scores)
        neg_loss = criterion(neg_scores, neg_labels)

        # Combine
        loss = (pos_loss + neg_loss) / 2.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_pos_loss += pos_loss.item()
        total_neg_loss += neg_loss.item()

    n_batches = len(loader)
    return (
        total_loss / n_batches,
        total_pos_loss / n_batches,
        total_neg_loss / n_batches,
    )

# -------------------------
# EVALUATION HELPERS
# -------------------------
@torch.no_grad()
def build_user_item_mappings(df, user2id, item2id):
    """
    Build dict: user_idx -> set(item_idx) for interactions in df (label == 1).
    """
    user_items = {}
    for _, row in df.iterrows():
        u = row["user_id"]
        i = row["parent_asin"]
        if u in user2id and i in item2id and row["label"] == 1:
            u_idx = user2id[u]
            i_idx = item2id[i]
            user_items.setdefault(u_idx, set()).add(i_idx)
    return user_items

@torch.no_grad()
def evaluate_ranking(model, test_df, user2id, item2id, k=10,
                     device=DEVICE, item_text_tensor=None):
    """
    Compute mean Recall@K, NDCG@K, HitRate@K over all users.
    """
    model.eval()
    user_pos_test = build_user_item_mappings(test_df, user2id, item2id)

    all_item_ids = torch.arange(len(item2id), dtype=torch.long, device=device)
    all_item_text = item_text_tensor[all_item_ids] if item_text_tensor is not None else None
    all_item_embs = model.get_item_embedding(all_item_ids, all_item_text)

    recalls, ndcgs, hitrates = [], [], []
    for u_idx, true_items in user_pos_test.items():
        if not true_items:
            continue

        u = torch.tensor([u_idx], dtype=torch.long, device=device)
        u_emb = model.get_user_embedding(u)
        scores = (u_emb @ all_item_embs.T).squeeze(0)
        _, topk_items = torch.topk(scores, k)
        ranked = topk_items.cpu().tolist()

        recalls.append(recall_at_k(true_items, ranked, k))
        ndcgs.append(ndcg_at_k(true_items, ranked, k))
        hitrates.append(hitrate_at_k(true_items, ranked, k))

    if len(recalls) == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(ndcgs)), float(np.mean(hitrates))

@torch.no_grad()
def evaluate_by_user_bucket(model, test_df, user2id, item2id, k=10,
                            device=DEVICE, item_text_tensor=None):
    """
    Compute metrics separately for cold/medium/hot users.
    """
    model.eval()
    user_pos_test = build_user_item_mappings(test_df, user2id, item2id)

    all_item_ids = torch.arange(len(item2id), dtype=torch.long, device=device)
    all_item_text = item_text_tensor[all_item_ids] if item_text_tensor is not None else None
    all_item_embs = model.get_item_embedding(all_item_ids, all_item_text)

    bucket_metrics = {
        "cold": {"rec": [], "ndcg": [], "hr": []},
        "medium": {"rec": [], "ndcg": [], "hr": []},
        "hot": {"rec": [], "ndcg": [], "hr": []},
    }

    for u_idx, true_items in user_pos_test.items():
        if not true_items:
            continue

        raw_u = id2user[u_idx]
        b = user_bucket(raw_u)

        u = torch.tensor([u_idx], dtype=torch.long, device=device)
        u_emb = model.get_user_embedding(u)
        scores = (u_emb @ all_item_embs.T).squeeze(0)
        _, topk_items = torch.topk(scores, k)
        ranked = topk_items.cpu().tolist()

        r = recall_at_k(true_items, ranked, k)
        n = ndcg_at_k(true_items, ranked, k)
        h = hitrate_at_k(true_items, ranked, k)

        bucket_metrics[b]["rec"].append(r)
        bucket_metrics[b]["ndcg"].append(n)
        bucket_metrics[b]["hr"].append(h)

    for b in bucket_metrics:
        for m in ["rec", "ndcg", "hr"]:
            vals = bucket_metrics[b][m]
            bucket_metrics[b][m] = float(np.mean(vals)) if vals else 0.0

    return bucket_metrics

# -------------------------
# POPULARITY BASELINE
# -------------------------
def build_popularity_ranking(train_df):
    pop_counter = Counter()
    for _, row in train_df.iterrows():
        if row["label"] == 1:
            pop_counter[row["parent_asin"]] += 1
    ranked_items = [asin for asin, _ in pop_counter.most_common()]
    return ranked_items

def evaluate_popularity_baseline(train_df, test_df, k=10):
    global_pop_rank = build_popularity_ranking(train_df)

    user_items = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["label"] == 1:
            user_items[row["user_id"]].add(row["parent_asin"])

    recalls, ndcgs, hitrates = [], [], []
    for user, true_items in user_items.items():
        if not true_items:
            continue
        ranked = global_pop_rank

        recalls.append(recall_at_k(true_items, ranked, k))
        ndcgs.append(ndcg_at_k(true_items, ranked, k))
        hitrates.append(hitrate_at_k(true_items, ranked, k))

    if len(recalls) == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(ndcgs)), float(np.mean(hitrates))

# -------------------------
# MAIN TRAINING LOOP + BEST-MODEL SELECTION + EVAL
# -------------------------
print("Evaluating popularity baseline on val...")
pop_rec, pop_ndcg, pop_hr = evaluate_popularity_baseline(train_df, val_df, k=10)
print(f"[Popularity] Val@10 | Recall: {pop_rec:.4f} | NDCG: {pop_ndcg:.4f} | HitRate: {pop_hr:.4f}")

best_val_recall = -1.0
best_epoch = -1
best_model_path = "best_model.pt"

print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, pos_loss, neg_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE, item_text_tensor=item_text_tensor
    )
    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Pos: {pos_loss:.4f} | "
        f"Neg: {neg_loss:.4f}"
    )

    rec, ndcg, hr = evaluate_ranking(
        model, val_df, user2id, item2id, k=10,
        device=DEVICE, item_text_tensor=item_text_tensor
    )
    print(f"  [Model] Val@10 | Recall: {rec:.4f} | NDCG: {ndcg:.4f} | HitRate: {hr:.4f}")

    # Save best model by Val Recall@10
    if rec > best_val_recall:
        best_val_recall = rec
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> New best model at epoch {epoch} (Val Recall@10 = {rec:.4f})")

# Load best model
print(f"Loading best model from epoch {best_epoch} with Val Recall@10 = {best_val_recall:.4f}")

if USE_ITEM_TEXT:
    best_model = TwoTowerWithItemText(
        num_users=num_users,
        num_items=num_items,
        user_emb_dim=EMBED_DIM,
        item_id_emb_dim=32,
        item_text_dim=ITEM_TEXT_DIM,
        out_dim=EMBED_DIM
    ).to(DEVICE)
else:
    best_model = TwoTowerIDOnly(num_users, num_items, emb_dim=EMBED_DIM).to(DEVICE)

best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

# Final test evaluation (best model)
rec_test, ndcg_test, hr_test = evaluate_ranking(
    best_model, test_df, user2id, item2id, k=10,
    device=DEVICE, item_text_tensor=item_text_tensor
)
print(f"[Best Model] Test@10 | Recall: {rec_test:.4f} | NDCG: {ndcg_test:.4f} | HitRate: {hr_test:.4f}")

bucket_res = evaluate_by_user_bucket(
    best_model, test_df, user2id, item2id, k=10,
    device=DEVICE, item_text_tensor=item_text_tensor
)
print("User-bucket test metrics (cold/medium/hot) for best model:")
print(bucket_res)
