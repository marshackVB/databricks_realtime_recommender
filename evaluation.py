# ---------------------------------------------------------------------------
# Evaluation metrics for the two-tower recommender model.
#
# Provides reusable functions for Recall@K, MRR, NDCG@K, popularity
# baseline, catalog coverage, blind spot analysis, and embedding geometry.
#
# Refactored to evaluate from Parquet files in a Unity Catalog volume, so the
# full training and evaluation sets never need to be loaded into pandas.
#
# The metrics need only three small lookups plus one streaming pass:
#   item_features  one row of item-tower features per training item
#   train_counts   training interaction count per item (popularity)
#   eval_items     the set of distinct items that appear in the eval set
#   eval_loader    batches of (user_batch, item_batch) read from Parquet
#
# Usage:
#   from evaluation import (
#       export_evaluation_inputs, load_evaluation_lookups, parquet_eval_batches,
#       run_full_evaluation, generate_evaluation_figures,
#   )
#
#   # 1. Once, in Spark, after building the training and eval Delta tables
#   export_evaluation_inputs(train_df, eval_df, VOLUME_PATH, eval_columns=EVAL_COLUMNS)
#
#   # 2. In the training notebook
#   item_features, train_counts, eval_items = load_evaluation_lookups(VOLUME_PATH)
#   eval_loader = parquet_eval_batches(f"{VOLUME_PATH}/eval", USER_COLUMNS, batch_size=2048)
#   metrics, details = run_full_evaluation(
#       model, item_features, train_counts, eval_items, eval_loader, device
#   )
#   figures = generate_evaluation_figures(metrics, details)
# ---------------------------------------------------------------------------
import glob
import math
import pickle

import numpy as np
import scipy.sparse as sp
import torch

# DataFrame column → ItemTower.forward() kwarg mapping
ITEM_EMB_COLUMNS = {
    "item_idx": ("item_id", torch.long),
    "category_idx": ("category_id", torch.long),
    "category_l2_idx": ("category_l2_id", torch.long),
    "category_l3_idx": ("category_l3_id", torch.long),
    "brand_idx": ("brand_id", torch.long),
    "item_log_price": ("log_price", torch.float32),
    "price_pct_rank": ("price_pct_rank", torch.float32),
    "price_pct_rank_l2": ("price_pct_rank_l2", torch.float32),
    "item_popularity": ("item_popularity", torch.float32),
    "item_conversion_rate": ("item_conversion_rate", torch.float32),
    "item_cart_to_purchase_rate": ("item_cart_to_purchase_rate", torch.float32),
    "item_recency_val": ("item_recency", torch.float32),
}

INTERACTION_BUCKETS = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, float("inf"))]


def _bucket_label(lo, hi):
    return f"{lo}+" if hi == float("inf") else f"{lo}-{hi}"


def _safe_label(label):
    return label.replace("+", "plus").replace("-", "_")


def _bucket_ids(counts, lo, hi):
    if hi == float("inf"):
        return set(counts[counts >= lo].index)
    return set(counts[(counts >= lo) & (counts <= hi)].index)


# ---- Data preparation (Spark) and loading ----


def export_evaluation_inputs(train_df, eval_df, volume_path, eval_columns,
                             time_col="event_time", eval_files=16):
    """Write everything evaluation needs to a Unity Catalog volume.

    Run once in Spark after the training and eval sets are built.

    Args:
        train_df: Spark DataFrame of the training set (features joined).
        eval_df: Spark DataFrame of the eval set (features joined).
        volume_path: e.g. "/Volumes/<catalog>/<schema>/<volume>/recsys".
        eval_columns: columns the eval loader reads (user-tower inputs + "item_idx").
        time_col: event timestamp column, used to pick each item's latest features.
        eval_files: number of Parquet files to write for the eval rows.

    Writes four folders under volume_path:
        eval/               eval rows, model columns only
        item_features/      one row per training item (latest feature values)
        train_item_counts/  item_idx, count
        eval_items/         distinct item_idx in the eval set
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    # Eval rows. No shuffle needed: row order does not affect the metrics.
    (eval_df.select(*eval_columns)
        .repartition(eval_files)
        .write.mode("overwrite").parquet(f"{volume_path}/eval"))

    # One row per item, using its most recent feature values in the training
    # period. Time-varying features (popularity, recency, ...) differ from row
    # to row, so "any row" is not good enough once the data is shuffled. The
    # latest row also matches what the serving index will hold.
    latest = Window.partitionBy("item_idx").orderBy(F.col(time_col).desc())
    (train_df.withColumn("_rn", F.row_number().over(latest))
        .filter(F.col("_rn") == 1)
        .select(*ITEM_EMB_COLUMNS.keys())
        .coalesce(1)
        .write.mode("overwrite").parquet(f"{volume_path}/item_features"))

    # Training interaction count per item (popularity baseline and buckets)
    (train_df.groupBy("item_idx").count()
        .coalesce(1)
        .write.mode("overwrite").parquet(f"{volume_path}/train_item_counts"))

    # Distinct items in the eval set (ground-truth coverage and blind spots)
    (eval_df.select("item_idx").distinct()
        .coalesce(1)
        .write.mode("overwrite").parquet(f"{volume_path}/eval_items"))


def load_evaluation_lookups(volume_path):
    """Load the three small lookups written by export_evaluation_inputs.

    Returns:
        item_features: DataFrame, one row per item, sorted by item_idx
        train_counts: Series indexed by item_idx, sorted by count descending
            (the same shape train_pdf["item_idx"].value_counts() used to return)
        eval_items: set of item_idx values present in the eval set
    """
    import pandas as pd

    item_features = (
        pd.read_parquet(f"{volume_path}/item_features")
        .sort_values("item_idx")
        .reset_index(drop=True)
    )
    counts = pd.read_parquet(f"{volume_path}/train_item_counts")
    train_counts = (
        counts.set_index("item_idx")["count"]
        .sort_values(ascending=False, kind="stable")
    )
    eval_items = set(pd.read_parquet(f"{volume_path}/eval_items")["item_idx"].tolist())
    return item_features, train_counts, eval_items


def _column_to_tensor(column, dtype):
    """Convert one Arrow column of a record batch to a torch tensor."""
    import pyarrow as pa

    col_type = column.type
    if (pa.types.is_list(col_type) or pa.types.is_large_list(col_type)
            or pa.types.is_fixed_size_list(col_type)):
        # Fixed-length list features (already padded in Spark) -> 2-D tensor
        values = np.asarray(column.to_pylist())
    else:
        values = column.to_numpy(zero_copy_only=False)
    return torch.tensor(values, dtype=dtype)


def parquet_eval_batches(path, user_columns, batch_size=2048, item_col="item_idx",
                         numeric_columns=None, numeric_kwarg="numeric"):
    """Stream eval batches from Parquet files in a volume.

    Args:
        path: folder of Parquet files (the "eval" folder from the export).
        user_columns: {parquet column: (UserTower.forward kwarg, torch dtype)},
            the user-tower equivalent of ITEM_EMB_COLUMNS.
        batch_size: rows per batch.
        item_col: column holding the true item index.
        numeric_columns: optional list of column names to stack into a single
            float32 (B, len(numeric_columns)) tensor under ``numeric_kwarg``.
        numeric_kwarg: kwarg name for the stacked numeric tensor.

    Yields:
        (user_batch, item_batch), where user_batch maps user-tower kwargs to
        tensors and item_batch is {"item_id": LongTensor}.  Memory use stays at
        one batch regardless of the size of the eval set.
    """
    import pyarrow.parquet as pq

    all_user_cols = list(user_columns.keys()) + (numeric_columns or [])
    columns = list(dict.fromkeys(all_user_cols + [item_col]))
    for file_path in sorted(glob.glob(f"{path}/*.parquet")):
        parquet_file = pq.ParquetFile(file_path)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            schema = record_batch.schema

            def get(name):
                return record_batch.column(schema.get_field_index(name))

            user_batch = {
                kwarg: _column_to_tensor(get(col), dtype)
                for col, (kwarg, dtype) in user_columns.items()
            }

            if numeric_columns:
                user_batch[numeric_kwarg] = torch.stack(
                    [_column_to_tensor(get(c), torch.float32)
                     for c in numeric_columns],
                    dim=-1,
                )

            item_batch = {"item_id": _column_to_tensor(get(item_col), torch.long)}
            yield user_batch, item_batch


# ---- Core evaluation functions ----


def build_item_embedding_index(model, item_features, device, batch_size=1024):
    """Build L2-normalized item embeddings for all unique training items.

    Args:
        item_features: pre-deduplicated DataFrame with one row per item
            (latest feature values, sorted by item_idx).

    Returns:
        item_embs: (num_items, embed_dim) tensor on device
        item_idx_list: numpy array mapping row position to item_idx
        item_features: the input DataFrame (passed through)
    """
    model.eval()
    cols = list(ITEM_EMB_COLUMNS.keys())
    item_features = item_features[cols].sort_values("item_idx").reset_index(drop=True)

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(item_features), batch_size):
            chunk = item_features.iloc[start : start + batch_size]
            kwargs = {
                kwarg: torch.tensor(chunk[col].values, dtype=dtype).to(device)
                for col, (kwarg, dtype) in ITEM_EMB_COLUMNS.items()
            }
            all_embeddings.append(model.item_tower(**kwargs))

    item_embs = torch.cat(all_embeddings, dim=0)
    return item_embs, item_features["item_idx"].values, item_features


def evaluate_single_pass(model, eval_loader, item_embs, item_idx_list, train_counts,
                         device, ks=(10, 50, 100, 150, 200),
                         coverage_ks=(10, 50, 100, 200),
                         item_features_df=None):
    """Compute retrieval metrics, the popularity baseline, and catalog
    coverage in ONE pass over the eval data.

    Replaces evaluate_retrieval_metrics, compute_popularity_baseline, and
    compute_catalog_coverage, which each looped over the eval loader.  The user
    tower and top-k now run once per batch, and all bookkeeping stays on the
    GPU, so eval_loader can be a one-shot generator such as
    parquet_eval_batches.

    Args:
        train_counts: Series indexed by item_idx, sorted by count descending.

    Returns:
        retrieval: dict with MRR, Recall@K, NDCG@K, random baselines
        pop: dict with the popularity-baseline MRR, Recall@K, NDCG@K
        coverage: dict with coverage_at_K and unique_retrieved_at_K
        retrieved_sets: {k: set of item_idx retrieved in any top-k}
        pop_ranked: numpy array of item_idx, most popular first
    """
    model.eval()
    ks = tuple(ks)
    coverage_ks = tuple(coverage_ks)
    catalog_size = len(item_idx_list)
    max_k = min(max(ks + coverage_ks), catalog_size)

    item_idx_np = np.asarray(item_idx_list)
    item_idx_t = torch.as_tensor(item_idx_np, dtype=torch.long, device=device)

    pop_ranked = np.asarray(train_counts.index.values)
    pop_idx_t = torch.as_tensor(pop_ranked, dtype=torch.long, device=device)
    default_rank = len(pop_ranked) + 1

    # Lookup tables indexed by item_idx:
    #   idx_to_pos  position of the item in item_embs, or -1 if not in the catalog
    #   pop_rank    1-based popularity rank, or default_rank if never seen in training
    size = int(max(item_idx_t.max().item(), pop_idx_t.max().item())) + 1
    idx_to_pos = torch.full((size,), -1, dtype=torch.long, device=device)
    idx_to_pos[item_idx_t] = torch.arange(catalog_size, device=device)
    pop_rank = torch.full((size,), default_rank, dtype=torch.long, device=device)
    pop_rank[pop_idx_t] = torch.arange(1, len(pop_ranked) + 1, device=device)

    # Within-category popularity rank tensor
    cat_pop_rank = None
    if item_features_df is not None and "category_idx" in item_features_df.columns:
        import pandas as pd
        _cat_map = item_features_df.set_index("item_idx")["category_idx"]
        _cpop = train_counts.reset_index()
        _cpop.columns = ["item_idx", "count"]
        _cpop["cat_idx"] = _cpop["item_idx"].map(_cat_map)
        _cpop["cat_rank"] = _cpop.groupby("cat_idx", dropna=False)["count"].rank(
            method="first", ascending=False
        ).astype(int)
        cat_pop_rank = torch.full((size,), default_rank, dtype=torch.long, device=device)
        _cpr_idx = torch.as_tensor(_cpop["item_idx"].values, dtype=torch.long, device=device)
        _cpr_vals = torch.as_tensor(_cpop["cat_rank"].values, dtype=torch.long, device=device)
        _valid = _cpr_idx < size
        cat_pop_rank[_cpr_idx[_valid]] = _cpr_vals[_valid]

    def _zero():
        return torch.zeros((), dtype=torch.float64, device=device)

    hits = {k: _zero() for k in ks}
    ndcg_sums = {k: _zero() for k in ks}
    rr_sum = _zero()
    pop_hits = {k: _zero() for k in ks}
    pop_ndcg_sums = {k: _zero() for k in ks}
    pop_rr_sum = _zero()
    cat_pop_hits = {k: _zero() for k in ks}
    cat_pop_ndcg_sums = {k: _zero() for k in ks}
    cat_pop_rr_sum = _zero()
    # Repeat vs novel split + recently-viewed baseline
    repeat_rr_sum = _zero()
    novel_rr_sum = _zero()
    recent_rr_sum = _zero()
    repeat_total = _zero()
    novel_total = _zero()
    repeat_hits = {k: _zero() for k in ks}
    novel_hits = {k: _zero() for k in ks}
    # One boolean flag per catalog item per K, instead of growing Python sets
    seen = {k: torch.zeros(catalog_size, dtype=torch.bool, device=device)
            for k in coverage_ks}
    total = 0

    with torch.no_grad():
        for user_batch, item_batch in eval_loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            true_idx = item_batch["item_id"].to(device).long()

            user_emb = model.user_tower(**user_batch)
            scores = user_emb @ item_embs.T
            topk_pos = scores.topk(max_k, dim=1).indices
            match_mask = item_idx_t[topk_pos] == true_idx.unsqueeze(1)

            # Recall@K and NDCG@K (one relevant item per row)
            for k in ks:
                mask_k = match_mask[:, :k]
                hit = mask_k.any(dim=1)
                rank_in_k = mask_k.float().argmax(dim=1)
                hits[k] += hit.sum()
                ndcg_sums[k] += (hit.double() / torch.log2(rank_in_k.double() + 2)).sum()

            # Items whose index is outside the lookup tables are unknown.
            # (The original clamped these onto the last training item, which
            # scored them as if they were that item.)
            in_range = (true_idx >= 0) & (true_idx < size)
            safe_idx = true_idx.clamp(0, size - 1)

            # MRR over the full catalog
            true_pos = idx_to_pos[safe_idx]
            valid = in_range & (true_pos >= 0)
            true_scores = scores.gather(1, true_pos.clamp(min=0).unsqueeze(1))
            ranks = (scores > true_scores).sum(dim=1) + 1
            rr_sum += (valid.double() / ranks.double()).sum()

            # Popularity baseline: always recommend the most-interacted items
            p_rank = torch.where(
                in_range, pop_rank[safe_idx], torch.full_like(true_idx, default_rank)
            )
            pop_rr_sum += (1.0 / p_rank.double()).sum()
            for k in ks:
                p_hit = p_rank <= k
                pop_hits[k] += p_hit.sum()
                pop_ndcg_sums[k] += (p_hit.double() / torch.log2(p_rank.double() + 1)).sum()

            # Category popularity baseline
            if cat_pop_rank is not None:
                cp_rank = torch.where(
                    in_range,
                    cat_pop_rank[safe_idx],
                    torch.full_like(true_idx, default_rank),
                )
                cat_pop_rr_sum += (1.0 / cp_rank.double()).sum()
                for k in ks:
                    cp_hit = cp_rank <= k
                    cat_pop_hits[k] += cp_hit.sum()
                    cat_pop_ndcg_sums[k] += (cp_hit.double() / torch.log2(cp_rank.double() + 1)).sum()

            # Repeat vs novel split + recently-viewed baseline
            _recent_ids = user_batch.get("recent_product_ids")
            _recent_mask = user_batch.get("recent_product_mask")
            if _recent_ids is not None and _recent_mask is not None:
                _match = (_recent_ids == true_idx.unsqueeze(1)) & (_recent_mask > 0)
                _is_repeat = _match.any(dim=1)
                _is_novel = ~_is_repeat

                _rr_vals = valid.double() / ranks.double()
                repeat_rr_sum += (_rr_vals * _is_repeat.double()).sum()
                novel_rr_sum += (_rr_vals * _is_novel.double()).sum()
                repeat_total += _is_repeat.sum()
                novel_total += _is_novel.sum()

                # Recently-viewed baseline: rank = position in recent list + 1
                _first_pos = _match.float().argmax(dim=1)
                _recent_rank = torch.where(
                    _is_repeat,
                    _first_pos.long() + 1,
                    torch.tensor(default_rank, dtype=torch.long, device=device),
                )
                recent_rr_sum += (1.0 / _recent_rank.double()).sum()

                # Split Recall@K
                for k in ks:
                    _hit_k = match_mask[:, :k].any(dim=1)
                    repeat_hits[k] += (_hit_k & _is_repeat).sum()
                    novel_hits[k] += (_hit_k & _is_novel).sum()

            # Catalog coverage
            for k in coverage_ks:
                seen[k][topk_pos[:, :k].reshape(-1)] = True

            total += true_idx.size(0)

    if total == 0:
        raise ValueError("eval_loader yielded no rows; check the Parquet path and columns.")

    retrieval = {
        "mrr": rr_sum.item() / total,
        "eval_interactions": total,
        "catalog_size": catalog_size,
    }
    pop = {"pop_mrr": pop_rr_sum.item() / total}
    cat_pop = {}
    for k in ks:
        retrieval[f"recall_at_{k}"] = hits[k].item() / total
        retrieval[f"ndcg_at_{k}"] = ndcg_sums[k].item() / total
        retrieval[f"random_recall_at_{k}"] = min(k / catalog_size, 1.0)
        pop[f"pop_recall_at_{k}"] = pop_hits[k].item() / total
        pop[f"pop_ndcg_at_{k}"] = pop_ndcg_sums[k].item() / total
    if cat_pop_rank is not None:
        cat_pop["cat_pop_mrr"] = cat_pop_rr_sum.item() / total
        for k in ks:
            cat_pop[f"cat_pop_recall_at_{k}"] = cat_pop_hits[k].item() / total
            cat_pop[f"cat_pop_ndcg_at_{k}"] = cat_pop_ndcg_sums[k].item() / total

    split = {}
    _rt = repeat_total.item()
    _nt = novel_total.item()
    if _rt + _nt > 0:
        split["repeat_mrr"] = repeat_rr_sum.item() / max(_rt, 1)
        split["novel_mrr"] = novel_rr_sum.item() / max(_nt, 1)
        split["recent_mrr"] = recent_rr_sum.item() / total
        split["repeat_pct"] = _rt / total
        split["novel_pct"] = _nt / total
        split["repeat_count"] = int(_rt)
        split["novel_count"] = int(_nt)
        for k in ks:
            split[f"repeat_recall_at_{k}"] = repeat_hits[k].item() / max(_rt, 1)
            split[f"novel_recall_at_{k}"] = novel_hits[k].item() / max(_nt, 1)

    coverage = {}
    retrieved_sets = {}
    for k in coverage_ks:
        retrieved_ids = item_idx_np[seen[k].cpu().numpy()]
        retrieved_sets[k] = set(retrieved_ids.tolist())
        coverage[f"coverage_at_{k}"] = len(retrieved_ids) / catalog_size
        coverage[f"unique_retrieved_at_{k}"] = int(len(retrieved_ids))

    return retrieval, pop, cat_pop, split, coverage, retrieved_sets, pop_ranked


def evaluate_masked_novel(model, eval_loader, item_embs, item_idx_list, train_counts,
                          device, ks=(10, 50, 100, 150, 200),
                          item_features_df=None):
    """Compute metrics with seen-item masking for novel interactions.

    Novel rows: items in recent_product_ids are masked from model scores
    (set to -inf) and removed from popularity rankings before computing
    MRR and Recall@K.  Sub-splits: has-history (recent_product_mask.sum > 0)
    vs first-event (no history).  Repeat rows use unmasked scores.

    Note on 'seen' coverage: recent_product_ids holds the last 10 distinct
    products within a 24-hour RollingWindow.  Products viewed outside that
    window count as novel.  This matches serving behaviour.

    Returns:
        dict of masked metrics keyed as masked_{slice}_{metric}.
    """
    model.eval()
    ks = tuple(ks)
    catalog_size = len(item_idx_list)
    max_k = min(max(ks), catalog_size)

    item_idx_np = np.asarray(item_idx_list)
    item_idx_t = torch.as_tensor(item_idx_np, dtype=torch.long, device=device)

    pop_ranked = np.asarray(train_counts.index.values)
    pop_idx_t = torch.as_tensor(pop_ranked, dtype=torch.long, device=device)
    default_rank = len(pop_ranked) + 1

    size = int(max(item_idx_t.max().item(), pop_idx_t.max().item())) + 1
    idx_to_pos = torch.full((size,), -1, dtype=torch.long, device=device)
    idx_to_pos[item_idx_t] = torch.arange(catalog_size, device=device)
    pop_rank = torch.full((size,), default_rank, dtype=torch.long, device=device)
    pop_rank[pop_idx_t] = torch.arange(1, len(pop_ranked) + 1, device=device)

    # Within-category popularity rank + item-to-category lookup
    cat_pop_rank, item_to_cat = None, None
    if item_features_df is not None and "category_idx" in item_features_df.columns:
        import pandas as pd
        _cat_map = item_features_df.set_index("item_idx")["category_idx"]
        _cpop = train_counts.reset_index()
        _cpop.columns = ["item_idx", "count"]
        _cpop["cat_idx"] = _cpop["item_idx"].map(_cat_map)
        _cpop["cat_rank"] = _cpop.groupby("cat_idx", dropna=False)["count"].rank(
            method="first", ascending=False
        ).astype(int)
        cat_pop_rank = torch.full((size,), default_rank, dtype=torch.long, device=device)
        _ci = torch.as_tensor(_cpop["item_idx"].values, dtype=torch.long, device=device)
        _cv = torch.as_tensor(_cpop["cat_rank"].values, dtype=torch.long, device=device)
        _ok = _ci < size
        cat_pop_rank[_ci[_ok]] = _cv[_ok]

        item_to_cat = torch.zeros(size, dtype=torch.long, device=device)
        _fi = torch.as_tensor(item_features_df["item_idx"].values, dtype=torch.long, device=device)
        _fc = torch.as_tensor(item_features_df["category_idx"].values, dtype=torch.long, device=device)
        _ok2 = _fi < size
        item_to_cat[_fi[_ok2]] = _fc[_ok2]

    def _zero():
        return torch.zeros((), dtype=torch.float64, device=device)

    _slices = ["novel", "novel_hist", "novel_first", "repeat"]
    mrr_acc = {s: _zero() for s in _slices}
    pop_mrr_acc = {s: _zero() for s in _slices}
    cat_mrr_acc = {s: _zero() for s in _slices}
    recall_acc = {s: {k: _zero() for k in ks} for s in _slices}
    pop_recall_acc = {s: {k: _zero() for k in ks} for s in _slices}
    cat_recall_acc = {s: {k: _zero() for k in ks} for s in _slices}
    totals = {s: _zero() for s in _slices}

    with torch.no_grad():
        for user_batch, item_batch in eval_loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            true_idx = item_batch["item_id"].to(device).long()

            user_emb = model.user_tower(**user_batch)
            scores = user_emb @ item_embs.T

            _recent_ids = user_batch.get("recent_product_ids")
            _recent_mask = user_batch.get("recent_product_mask")
            if _recent_ids is None or _recent_mask is None:
                continue

            # Classify rows
            _match = (_recent_ids == true_idx.unsqueeze(1)) & (_recent_mask > 0)
            _is_repeat = _match.any(dim=1)
            _is_novel = ~_is_repeat
            _has_hist = _recent_mask.sum(dim=1) > 0
            _is_nh = _is_novel & _has_hist
            _is_nf = _is_novel & ~_has_hist

            totals["repeat"] += _is_repeat.sum()
            totals["novel"] += _is_novel.sum()
            totals["novel_hist"] += _is_nh.sum()
            totals["novel_first"] += _is_nf.sum()

            # Seen-item mask: map recent item indices to score-matrix columns.
            # Guard padding (recent_product_mask == 0) and items not in the
            # catalog (idx_to_pos == -1).  idx_to_pos is -1 for unknown items;
            # without this guard we would mask column -1 (the last column),
            # silently removing a valid product from every row.
            _seen_idx = _recent_ids.clamp(0, size - 1)
            _seen_pos = idx_to_pos[_seen_idx]
            _seen_ok = (_recent_mask > 0) & (_seen_pos >= 0)

            masked_scores = scores.clone()
            for j in range(_recent_ids.size(1)):
                _m = _seen_ok[:, j] & _is_novel
                if _m.any():
                    masked_scores[_m, _seen_pos[:, j][_m]] = float("-inf")

            # Shared lookups
            in_range = (true_idx >= 0) & (true_idx < size)
            safe_idx = true_idx.clamp(0, size - 1)
            true_pos = idx_to_pos[safe_idx]
            valid = in_range & (true_pos >= 0)

            # -- Model MRR (masked for novel, unmasked for repeat) --
            m_ts = masked_scores.gather(1, true_pos.clamp(min=0).unsqueeze(1))
            m_ranks = (masked_scores > m_ts).sum(dim=1) + 1
            m_rr = valid.double() / m_ranks.double()

            r_ts = scores.gather(1, true_pos.clamp(min=0).unsqueeze(1))
            r_ranks = (scores > r_ts).sum(dim=1) + 1
            r_rr = valid.double() / r_ranks.double()

            mrr_acc["repeat"] += (r_rr * _is_repeat.double()).sum()
            mrr_acc["novel"] += (m_rr * _is_novel.double()).sum()
            mrr_acc["novel_hist"] += (m_rr * _is_nh.double()).sum()
            mrr_acc["novel_first"] += (m_rr * _is_nf.double()).sum()

            # -- Model Recall@K --
            m_topk = masked_scores.topk(max_k, dim=1).indices
            m_hit = item_idx_t[m_topk] == true_idx.unsqueeze(1)
            r_topk = scores.topk(max_k, dim=1).indices
            r_hit = item_idx_t[r_topk] == true_idx.unsqueeze(1)

            for k in ks:
                nh = m_hit[:, :min(k, max_k)].any(dim=1)
                rh = r_hit[:, :min(k, max_k)].any(dim=1)
                recall_acc["repeat"][k] += (rh & _is_repeat).sum()
                recall_acc["novel"][k] += (nh & _is_novel).sum()
                recall_acc["novel_hist"][k] += (nh & _is_nh).sum()
                recall_acc["novel_first"][k] += (nh & _is_nf).sum()

            # -- Masked global-popularity baseline --
            tgt_pr = torch.where(
                in_range, pop_rank[safe_idx],
                torch.full_like(true_idx, default_rank),
            )
            _seen_pr = pop_rank[_seen_idx]
            _better_pop = (_seen_ok & (_seen_pr < tgt_pr.unsqueeze(1))).sum(dim=1)
            msk_pr = (tgt_pr - _better_pop).clamp(min=1)

            pop_mrr_acc["repeat"] += (_is_repeat.double() / tgt_pr.double()).sum()
            pop_mrr_acc["novel"] += (_is_novel.double() / msk_pr.double()).sum()
            pop_mrr_acc["novel_hist"] += (_is_nh.double() / msk_pr.double()).sum()
            pop_mrr_acc["novel_first"] += (_is_nf.double() / msk_pr.double()).sum()

            for k in ks:
                pop_recall_acc["repeat"][k] += ((tgt_pr <= k) & _is_repeat).sum()
                pop_recall_acc["novel"][k] += ((msk_pr <= k) & _is_novel).sum()
                pop_recall_acc["novel_hist"][k] += ((msk_pr <= k) & _is_nh).sum()
                pop_recall_acc["novel_first"][k] += ((msk_pr <= k) & _is_nf).sum()

            # -- Masked category-popularity baseline --
            if cat_pop_rank is not None and item_to_cat is not None:
                tgt_cr = torch.where(
                    in_range, cat_pop_rank[safe_idx],
                    torch.full_like(true_idx, default_rank),
                )
                tgt_cat = item_to_cat[safe_idx]
                _seen_cat = item_to_cat[_seen_idx]
                _seen_cr = cat_pop_rank[_seen_idx]
                _same_cat = (_seen_cat == tgt_cat.unsqueeze(1)) & _seen_ok
                _better_cp = (_same_cat & (_seen_cr < tgt_cr.unsqueeze(1))).sum(dim=1)
                msk_cr = (tgt_cr - _better_cp).clamp(min=1)

                cat_mrr_acc["repeat"] += (_is_repeat.double() / tgt_cr.double()).sum()
                cat_mrr_acc["novel"] += (_is_novel.double() / msk_cr.double()).sum()
                cat_mrr_acc["novel_hist"] += (_is_nh.double() / msk_cr.double()).sum()
                cat_mrr_acc["novel_first"] += (_is_nf.double() / msk_cr.double()).sum()

                for k in ks:
                    cat_recall_acc["repeat"][k] += ((tgt_cr <= k) & _is_repeat).sum()
                    cat_recall_acc["novel"][k] += ((msk_cr <= k) & _is_novel).sum()
                    cat_recall_acc["novel_hist"][k] += ((msk_cr <= k) & _is_nh).sum()
                    cat_recall_acc["novel_first"][k] += ((msk_cr <= k) & _is_nf).sum()

    result = {}
    for s in _slices:
        t = totals[s].item()
        if t == 0:
            continue
        p = f"masked_{s}"
        result[f"{p}_count"] = int(t)
        result[f"{p}_mrr"] = mrr_acc[s].item() / t
        result[f"{p}_pop_mrr"] = pop_mrr_acc[s].item() / t
        if cat_pop_rank is not None:
            result[f"{p}_cat_pop_mrr"] = cat_mrr_acc[s].item() / t
        for k in ks:
            result[f"{p}_recall_at_{k}"] = recall_acc[s][k].item() / t
            result[f"{p}_pop_recall_at_{k}"] = pop_recall_acc[s][k].item() / t
            if cat_pop_rank is not None:
                result[f"{p}_cat_pop_recall_at_{k}"] = cat_recall_acc[s][k].item() / t

    return result


def load_cooccurrence_matrix(volume_path):
    """Load the cosine-normalized co-occurrence matrix from the volume."""
    path = f"{volume_path}/cooc_matrix.pkl"
    with open(path, "rb") as f:
        matrix = pickle.load(f)
    print(f"Loaded co-occurrence matrix: {matrix.shape}, {matrix.nnz:,} nnz")
    return matrix


def evaluate_cooccurrence_baseline(
    eval_path, cooc_matrix, item_features_df, batch_size=2048,
    ks=(10, 50, 100, 150, 200),
):
    """Evaluate a session co-occurrence baseline over eval Parquet files.

    For each eval row with session history, scores every candidate item by
    the sum of cosine-normalized co-occurrence with the user's recent products.
    Seen items are masked (set to -inf) for novel rows.  Repeat rows use
    unmasked scores.  First-event rows (no recent products) are reported
    separately with rank = catalog_size (co-occurrence is undefined).

    Runs entirely on CPU — no model or GPU needed.

    Returns:
        dict of ``cooc_{slice}_{metric}`` keys.
    """
    import pyarrow.parquet as _pq

    cooc_size = cooc_matrix.shape[0]
    eval_files = sorted(glob.glob(f"{eval_path}/*.parquet"))
    needed_cols = ["item_idx", "recent_product_ids", "recent_product_mask"]

    # Build item_idx → category_l1 mapping for cat-pop fallback on first-event
    item_to_cat = None
    if item_features_df is not None and "category_idx" in item_features_df.columns:
        _cat = item_features_df.set_index("item_idx")["category_idx"]
        max_idx = max(int(_cat.index.max()), cooc_size - 1)
        item_to_cat = np.zeros(max_idx + 1, dtype=np.int32)
        item_to_cat[_cat.index.values] = _cat.values.astype(np.int32)

    # Accumulators per slice
    slices = ["novel_hist", "novel_first", "novel", "repeat"]
    mrr_acc = {s: 0.0 for s in slices}
    recall_acc = {s: {k: 0 for k in ks} for s in slices}
    count_acc = {s: 0 for s in slices}

    total_rows = 0
    for fi, fpath in enumerate(eval_files):
        pf = _pq.ParquetFile(fpath)
        for rb in pf.iter_batches(batch_size, columns=needed_cols):
            item_idx = rb.column("item_idx").to_numpy(zero_copy_only=False)
            rp = np.stack(rb.column("recent_product_ids").to_pandas().values)
            rm = np.stack(rb.column("recent_product_mask").to_pandas().values)
            B = len(item_idx)
            total_rows += B

            # Classify rows
            has_hist = rm.sum(axis=1) > 0
            is_repeat = np.zeros(B, dtype=bool)
            for j in range(rp.shape[1]):
                is_repeat |= ((rp[:, j] == item_idx) & (rm[:, j] > 0))
            is_novel = ~is_repeat
            is_nh = is_novel & has_hist    # novel with history
            is_nf = is_novel & ~has_hist   # first-event

            # --- Co-occurrence scoring for rows WITH history ---
            hist_idx = np.where(has_hist)[0]
            if len(hist_idx) > 0:
                _rp_h = rp[hist_idx]
                _rm_h = rm[hist_idx]
                _ti_h = np.clip(item_idx[hist_idx], 0, cooc_size - 1)
                B_h = len(hist_idx)

                # Build sparse indicator (B_h, cooc_size)
                _ri, _ci = [], []
                for i in range(B_h):
                    for j in range(_rp_h.shape[1]):
                        idx = int(_rp_h[i, j])
                        if _rm_h[i, j] > 0 and 0 < idx < cooc_size:
                            _ri.append(i)
                            _ci.append(idx)
                _indicator = sp.csr_matrix(
                    (np.ones(len(_ri), dtype=np.float32), (_ri, _ci)),
                    shape=(B_h, cooc_size),
                )

                # Sparse multiply
                result_csr = (_indicator @ cooc_matrix)
                result_csr.sort_indices()

                # Target scores
                tgt_sc = np.array(result_csr[np.arange(B_h), _ti_h]).flatten()

                # Vectorized ranking from CSR structure with expected-rank
                # tie-breaking.  Co-occurrence scores are integers so many
                # candidates share the same score (especially 0).  We use:
                #   rank = n_greater + 1 + n_tied / 2
                # where n_tied counts items with the same score as the target,
                # excluding the target itself.  This gives the expected rank
                # under random tie-breaking and removes the inflation at
                # higher K that favoured zero-scored targets.
                indptr = result_csr.indptr
                data = result_csr.data
                nnz_per_row = np.diff(indptr).astype(np.int64)
                row_for_entry = np.repeat(np.arange(B_h), nnz_per_row)

                above = data > tgt_sc[row_for_entry]
                above_counts = np.bincount(
                    row_for_entry[above], minlength=B_h
                ).astype(np.float64)

                # Count tied items (same score as target, excluding target)
                tied = data == tgt_sc[row_for_entry]
                tied_counts = np.bincount(
                    row_for_entry[tied], minlength=B_h
                ).astype(np.float64)
                # If target score > 0 it appears in CSR and was counted — remove it
                tied_counts -= (tgt_sc > 0).astype(np.float64)
                # For zero-scored targets, add all other zero-scored items
                # (they are not stored in the CSR)
                tied_counts += (tgt_sc == 0) * (cooc_size - nnz_per_row - 1)

                # Subtract seen items that score above OR are tied with
                # target (novel rows only — seen items are masked out)
                _is_novel_h = is_novel[hist_idx]
                for j in range(_rp_h.shape[1]):
                    valid = (_rm_h[:, j] > 0) & _is_novel_h
                    valid_rows = np.where(valid)[0]
                    if len(valid_rows) == 0:
                        continue
                    seen_idx = np.clip(
                        _rp_h[valid, j].astype(int), 0, cooc_size - 1
                    )
                    seen_scores = np.array(
                        result_csr[valid_rows, seen_idx]
                    ).flatten()
                    is_above = seen_scores > tgt_sc[valid]
                    np.subtract.at(above_counts, valid_rows[is_above], 1)
                    is_tied = seen_scores == tgt_sc[valid]
                    np.subtract.at(tied_counts, valid_rows[is_tied], 1)

                # Defensive clamp (shouldn't be needed but guards edge cases)
                tied_counts = np.maximum(tied_counts, 0.0)

                # Expected rank under random tie-breaking
                ranks_h = above_counts + 1.0 + tied_counts / 2.0
                rr_h = 1.0 / ranks_h

                # Map back to full-batch slice masks
                _is_rep_h = is_repeat[hist_idx]
                _is_nh_h = is_nh[hist_idx]

                mrr_acc["repeat"] += (rr_h * _is_rep_h).sum()
                mrr_acc["novel_hist"] += (rr_h * _is_nh_h).sum()
                mrr_acc["novel"] += (rr_h * _is_nh_h).sum()  # novel = nh + nf

                count_acc["repeat"] += int(_is_rep_h.sum())
                count_acc["novel_hist"] += int(_is_nh_h.sum())

                for k in ks:
                    hit = ranks_h <= k
                    recall_acc["repeat"][k] += int((hit & _is_rep_h).sum())
                    recall_acc["novel_hist"][k] += int((hit & _is_nh_h).sum())
                    recall_acc["novel"][k] += int((hit & _is_nh_h).sum())

            # --- First-event rows: rank = cooc_size (undefined) ---
            n_nf = int(is_nf.sum())
            count_acc["novel_first"] += n_nf
            count_acc["novel"] += n_nf
            # MRR and recall stay at 0 for first-event (co-occurrence undefined)

        if (fi + 1) % 5 == 0 or fi == len(eval_files) - 1:
            print(f"  Co-occurrence eval: {fi+1}/{len(eval_files)} files, "
                  f"{total_rows:,} rows processed")

    # Compile results
    result = {}
    for s in slices:
        t = count_acc[s]
        if t == 0:
            continue
        p = f"cooc_{s}"
        result[f"{p}_count"] = t
        result[f"{p}_mrr"] = mrr_acc[s] / t
        for k in ks:
            result[f"{p}_recall_at_{k}"] = recall_acc[s][k] / t

    print(f"  Co-occurrence eval complete: {total_rows:,} rows")
    return result


def analyze_retrieval_vs_ground_truth(eval_items, retrieved_sets, item_idx_list,
                                      ks=(100, 200)):
    """Coverage vs eval ground truth and blind spot identification.

    Args:
        eval_items: set of item_idx values that appear in the eval set.
    """
    train_catalog = set(item_idx_list.tolist())
    cold_start = eval_items - train_catalog

    metrics = {
        "eval_distinct_items": len(eval_items),
        "cold_start_items": len(cold_start),
        "cold_start_pct": len(cold_start) / max(len(eval_items), 1),
    }
    blind_spots, warm_blind = {}, {}
    for k in ks:
        if k not in retrieved_sets:
            continue
        covered = retrieved_sets[k] & eval_items
        blind_k = eval_items - retrieved_sets[k]
        warm_k = blind_k - cold_start
        metrics[f"gt_coverage_at_{k}"] = len(covered) / max(len(eval_items), 1)
        metrics[f"blind_spots_at_{k}"] = len(blind_k)
        metrics[f"blind_spot_pct_at_{k}"] = len(blind_k) / max(len(eval_items), 1)
        metrics[f"warm_blind_at_{k}"] = len(warm_k)
        blind_spots[k] = blind_k
        warm_blind[k] = warm_k

    return metrics, blind_spots, warm_blind, cold_start, train_catalog


def analyze_blind_spots(train_counts, blind_spot_ids, retrieved_ids, eval_items):
    """Blind spot distribution by training interaction count.

    Args:
        train_counts: pandas Series indexed by item_idx with training
            interaction counts.
        eval_items: set of distinct item_idx values in the eval set.

    Reports both catalog-basis and eval-conditioned miss rates to avoid
    denominator artifacts (per evaluation methodology: the covariate
    'training frequency' also drives eval-set eligibility).
    """
    bs_counts = train_counts[train_counts.index.isin(blind_spot_ids)]
    ret_counts = train_counts[train_counts.index.isin(retrieved_ids)]

    metrics = {
        "bs_median_interactions": float(bs_counts.median()) if len(bs_counts) else 0,
        "bs_mean_interactions": float(bs_counts.mean()) if len(bs_counts) else 0,
        "ret_median_interactions": float(ret_counts.median()) if len(ret_counts) else 0,
        "ret_mean_interactions": float(ret_counts.mean()) if len(ret_counts) else 0,
    }

    bucket_analysis = []
    for lo, hi in INTERACTION_BUCKETS:
        label = _bucket_label(lo, hi)
        cat_ids = _bucket_ids(train_counts, lo, hi)
        cat_n = len(cat_ids)
        eval_ids = cat_ids & eval_items
        eval_n = len(eval_ids)
        blind_n = len(eval_ids & blind_spot_ids)
        miss_eval = blind_n / max(eval_n, 1)
        miss_cat = blind_n / max(cat_n, 1)

        bucket_analysis.append({
            "bucket": label, "catalog": cat_n, "in_eval": eval_n,
            "blind": blind_n, "miss_rate_eval": miss_eval, "miss_rate_cat": miss_cat,
        })
        safe = _safe_label(label)
        metrics[f"miss_eval_{safe}"] = miss_eval
        metrics[f"miss_cat_{safe}"] = miss_cat

    return metrics, bucket_analysis


def analyze_embedding_geometry(item_embs_np, item_idx_list, item_features_df,
                               all_train_counts, blind_spot_ids, rng_seed=42):
    """Embedding displacement hypothesis: cosine similarity and category
    diversity by interaction frequency bucket."""
    rng = np.random.default_rng(rng_seed)
    idx_to_row = {int(idx): pos for pos, idx in enumerate(item_idx_list)}

    def _bootstrap_cos(embs, repeats=40, sample_size=512):
        if len(embs) < 2:
            return np.nan, np.nan, np.nan
        take_n = min(len(embs), sample_size)
        tri = np.triu_indices(take_n, k=1)
        means = []
        for _ in range(repeats):
            rows = rng.choice(len(embs), size=take_n, replace=False)
            cos = embs[rows] @ embs[rows].T
            means.append(float(cos[tri].mean()))
        m = float(np.mean(means))
        lo, hi = np.percentile(means, [2.5, 97.5])
        return m, float(lo), float(hi)

    bucket_results = {}
    for blo, bhi in INTERACTION_BUCKETS:
        label = _bucket_label(blo, bhi)
        ids = _bucket_ids(all_train_counts, blo, bhi)
        rows = [idx_to_row[i] for i in ids if i in idx_to_row]
        if len(rows) < 2:
            continue

        embs = item_embs_np[rows]
        mean_cos, ci_lo, ci_hi = _bootstrap_cos(embs)

        # Category diversity (L2)
        bf = item_features_df[item_features_df["item_idx"].isin(ids)]
        if "category_l2_idx" in bf.columns and len(bf) > 0:
            l2_probs = bf["category_l2_idx"].value_counts(normalize=True)
            l2_count = int(bf["category_l2_idx"].nunique())
            l2_entropy = float(
                -(l2_probs * np.log(l2_probs + 1e-12)).sum()
                / np.log(max(len(l2_probs), 2))
            )
        else:
            l2_count, l2_entropy = 0, 0.0

        blind_n = len(set(list(ids)) & blind_spot_ids)
        miss = blind_n / max(len(ids), 1)

        bucket_results[label] = {
            "n": len(rows), "mean_cos": mean_cos, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "l2_count": l2_count, "l2_entropy": l2_entropy, "miss_rate": miss,
        }

    metrics = {}
    for label, res in bucket_results.items():
        safe = _safe_label(label)
        metrics[f"emb_cos_{safe}"] = res["mean_cos"]
        metrics[f"emb_l2_entropy_{safe}"] = res["l2_entropy"]
    return metrics, bucket_results


# ---- Orchestrator ----


def run_full_evaluation(model, item_features, train_counts, eval_items, eval_loader,
                        device, ks=(10, 50, 100, 150, 200),
                        coverage_ks=(10, 50, 100, 200),
                        eval_loader_fn=None,
                        eval_path=None, cooc_matrix=None):
    """Run all evaluation metrics.

    Args:
        item_features: DataFrame, one row per training item (ITEM_EMB_COLUMNS).
        train_counts: Series indexed by item_idx with training interaction counts.
        eval_items: set of distinct item_idx values in the eval set.
        eval_loader: iterable of (user_batch, item_batch).  It is consumed once,
            so a generator such as parquet_eval_batches works.
        eval_loader_fn: optional callable returning a fresh eval_loader for a
            second pass (masked novel evaluation).  If None, masked eval is
            skipped.

    Returns:
        metrics: flat dict of all numeric metrics (ready for mlflow.log_metrics)
        details: dict of detailed objects for downstream use / artifact logging
    """
    train_counts = train_counts.sort_values(ascending=False, kind="stable")
    eval_items = set(eval_items)

    print("Building item embedding index...")
    item_embs, item_idx_list, item_features = build_item_embedding_index(
        model, item_features, device
    )

    print("Computing Recall@K, MRR, NDCG@K, popularity baseline, and coverage...")
    retrieval, pop, cat_pop, split, cov, retrieved_sets, pop_ranked = evaluate_single_pass(
        model, eval_loader, item_embs, item_idx_list, train_counts, device,
        ks=ks, coverage_ks=coverage_ks, item_features_df=item_features,
    )

    # Lift over popularity
    lifts = {}
    for k in ks:
        lifts[f"recall_lift_at_{k}"] = retrieval[f"recall_at_{k}"] / max(pop[f"pop_recall_at_{k}"], 1e-10)
        lifts[f"ndcg_lift_at_{k}"] = retrieval[f"ndcg_at_{k}"] / max(pop[f"pop_ndcg_at_{k}"], 1e-10)
    lifts["mrr_lift"] = retrieval["mrr"] / max(pop["pop_mrr"], 1e-10)

    # Category-level lift
    cat_lifts = {}
    if cat_pop:
        for k in ks:
            cat_lifts[f"cat_recall_lift_at_{k}"] = retrieval[f"recall_at_{k}"] / max(cat_pop.get(f"cat_pop_recall_at_{k}", 1e-10), 1e-10)
        cat_lifts["cat_mrr_lift"] = retrieval["mrr"] / max(cat_pop.get("cat_pop_mrr", 1e-10), 1e-10)

    # Popularity concentration in top-100
    if 100 in retrieved_sets:
        top1k = set(pop_ranked[:1000].tolist())
        overlap = retrieved_sets[100] & top1k
        cov["pop_concentration_100"] = len(overlap) / max(len(retrieved_sets[100]), 1)

    print("Analyzing retrieval vs ground truth...")
    gt, blind_spots, warm_blind, cold_start, train_catalog = (
        analyze_retrieval_vs_ground_truth(eval_items, retrieved_sets, item_idx_list)
    )

    print("Analyzing blind spots...")
    blind_200 = warm_blind.get(200, set())
    retrieved_200 = retrieved_sets.get(200, set()) & train_catalog
    bs, bucket_analysis = analyze_blind_spots(
        train_counts, blind_200, retrieved_200, eval_items
    )

    print("Analyzing embedding geometry...")
    emb, bucket_results = analyze_embedding_geometry(
        item_embs.cpu().numpy(), item_idx_list, item_features,
        train_counts, blind_200
    )

    # Merge all metrics
    metrics = {}
    for d in [retrieval, pop, cat_pop, split, lifts, cat_lifts, cov, gt, bs, emb]:
        metrics.update(d)

    details = {
        "item_embs": item_embs,
        "item_idx_list": item_idx_list,
        "item_features": item_features,
        "pop_ranked": pop_ranked,
        "retrieved_sets": retrieved_sets,
        "blind_spots": blind_spots,
        "warm_blind": warm_blind,
        "cold_start_items": cold_start,
        "bucket_analysis": bucket_analysis,
        "embedding_bucket_results": bucket_results,
    }

    # Masked novel evaluation (second pass)
    if eval_loader_fn is not None:
        print("Computing masked novel metrics (second pass)...")
        masked_loader = eval_loader_fn()
        masked = evaluate_masked_novel(
            model, masked_loader, item_embs, item_idx_list, train_counts,
            device, ks=ks, item_features_df=item_features,
        )
        metrics.update(masked)

    # Co-occurrence baseline (CPU-only third pass)
    if cooc_matrix is not None and eval_path is not None:
        print("Computing co-occurrence baseline (CPU pass)...")
        cooc_metrics = evaluate_cooccurrence_baseline(
            eval_path, cooc_matrix, item_features,
            batch_size=2048, ks=ks,
        )
        metrics.update(cooc_metrics)

    print("Evaluation complete.\n")
    return metrics, details


# ---- Figures ----


def generate_evaluation_figures(metrics, details, ks=(10, 50, 100, 150, 200)):
    """Generate matplotlib figures for MLflow artifact logging."""
    import matplotlib.pyplot as plt

    figures = {}

    # 1. Model vs Popularity: Recall and NDCG side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = list(range(len(ks)))
    w = 0.35
    model_r = [metrics.get(f"recall_at_{k}", 0) for k in ks]
    pop_r = [metrics.get(f"pop_recall_at_{k}", 0) for k in ks]
    ax1.bar([i - w / 2 for i in x], model_r, w, label="Model", color="steelblue")
    ax1.bar([i + w / 2 for i in x], pop_r, w, label="Popularity", color="lightcoral")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel("K")
    ax1.set_ylabel("Recall@K")
    ax1.legend()
    ax1.set_title("Recall@K: Model vs Popularity")

    model_n = [metrics.get(f"ndcg_at_{k}", 0) for k in ks]
    pop_n = [metrics.get(f"pop_ndcg_at_{k}", 0) for k in ks]
    ax2.bar([i - w / 2 for i in x], model_n, w, label="Model", color="steelblue")
    ax2.bar([i + w / 2 for i in x], pop_n, w, label="Popularity", color="lightcoral")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("K")
    ax2.set_ylabel("NDCG@K")
    ax2.legend()
    ax2.set_title("NDCG@K: Model vs Popularity")
    plt.tight_layout()
    figures["model_vs_popularity"] = fig

    # 2. Catalog coverage
    cov_ks = sorted(
        int(k.split("_")[-1])
        for k in metrics
        if k.startswith("coverage_at_")
    )
    if cov_ks:
        fig2, ax = plt.subplots(figsize=(8, 4))
        vals = [metrics[f"coverage_at_{k}"] * 100 for k in cov_ks]
        ax.bar(range(len(cov_ks)), vals, color="steelblue")
        ax.set_xticks(range(len(cov_ks)))
        ax.set_xticklabels([f"Top-{k}" for k in cov_ks])
        ax.set_ylabel("% of Catalog")
        ax.set_title("Catalog Coverage")
        plt.tight_layout()
        figures["catalog_coverage"] = fig2

    # 3. Blind spot miss rate by interaction bucket (eval-conditioned)
    bucket_analysis = details.get("bucket_analysis", [])
    if bucket_analysis:
        fig3, ax = plt.subplots(figsize=(8, 4))
        labels = [b["bucket"] for b in bucket_analysis]
        miss = [b["miss_rate_eval"] * 100 for b in bucket_analysis]
        ax.bar(range(len(labels)), miss, color="coral")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Training Interaction Count")
        ax.set_ylabel("Miss Rate (eval-conditioned, %)")
        ax.set_title("Blind Spot Miss Rate by Frequency")
        plt.tight_layout()
        figures["blind_spot_miss_rates"] = fig3

    # 4. Embedding self-similarity by interaction bucket
    emb_buckets = details.get("embedding_bucket_results", {})
    if emb_buckets:
        fig4, ax = plt.subplots(figsize=(8, 4))
        blabels = list(emb_buckets.keys())
        cos_means = [emb_buckets[l]["mean_cos"] for l in blabels]
        ci_lo = [emb_buckets[l]["ci_lo"] for l in blabels]
        ci_hi = [emb_buckets[l]["ci_hi"] for l in blabels]
        yerr = [[m - lo for m, lo in zip(cos_means, ci_lo)],
                [hi - m for m, hi in zip(cos_means, ci_hi)]]
        ax.bar(range(len(blabels)), cos_means, yerr=yerr, capsize=4, color="mediumpurple")
        ax.set_xticks(range(len(blabels)))
        ax.set_xticklabels(blabels)
        ax.set_xlabel("Training Interaction Count")
        ax.set_ylabel("Mean Pairwise Cosine")
        ax.set_title("Embedding Self-Similarity (95% CI)")
        plt.tight_layout()
        figures["embedding_geometry"] = fig4

    return figures


# ---- Summary printer ----


def print_evaluation_summary(metrics, details):
    """Print formatted evaluation results to stdout."""
    ks = sorted(
        int(k.split("_")[-1]) for k in metrics if k.startswith("recall_at_")
    )

    print(f"\n{'='*65}")
    print(f"  RETRIEVAL METRICS ({metrics.get('eval_interactions', 0):,} eval interactions)")
    print(f"{'='*65}")
    has_cat = "cat_pop_mrr" in metrics
    if has_cat:
        print(f"  {'Metric':<18} {'Model':>10} {'Global Pop':>12} {'Cat Pop':>10} {'Lift':>8} {'Cat Lift':>10}")
        print(f"  {'-'*70}")
        print(
            f"  {'MRR':<18} {metrics['mrr']:>10.4f} "
            f"{metrics['pop_mrr']:>12.4f} {metrics['cat_pop_mrr']:>10.4f} "
            f"{metrics['mrr_lift']:>7.1f}x {metrics['cat_mrr_lift']:>9.1f}x"
        )
        for k in ks:
            mr = metrics[f"recall_at_{k}"]
            pr = metrics[f"pop_recall_at_{k}"]
            cr = metrics[f"cat_pop_recall_at_{k}"]
            lr = metrics[f"recall_lift_at_{k}"]
            clr = metrics[f"cat_recall_lift_at_{k}"]
            print(f"  {'Recall@' + str(k):<18} {mr:>10.4f} {pr:>12.4f} {cr:>10.4f} {lr:>7.1f}x {clr:>9.1f}x")
    else:
        print(f"  {'Metric':<18} {'Model':>10} {'Popularity':>12} {'Lift':>8}")
        print(f"  {'-'*48}")
        print(
            f"  {'MRR':<18} {metrics['mrr']:>10.4f} "
            f"{metrics['pop_mrr']:>12.4f} {metrics['mrr_lift']:>7.1f}x"
        )
        for k in ks:
            mr = metrics[f"recall_at_{k}"]
            pr = metrics[f"pop_recall_at_{k}"]
            lr = metrics[f"recall_lift_at_{k}"]
            print(f"  {'Recall@' + str(k):<18} {mr:>10.4f} {pr:>12.4f} {lr:>7.1f}x")

    # Coverage
    cov_ks = sorted(
        int(k.split("_")[-1]) for k in metrics if k.startswith("coverage_at_")
    )
    if cov_ks:
        print(f"\n  CATALOG COVERAGE (catalog: {metrics.get('catalog_size', 0):,} items)")
        print(f"  {'-'*40}")
        for k in cov_ks:
            n = metrics.get(f"unique_retrieved_at_{k}", 0)
            pct = metrics[f"coverage_at_{k}"] * 100
            print(f"  Top-{k:<4d}: {n:>6,} unique items = {pct:>5.1f}%")

    # Ground truth coverage
    print(f"\n  GROUND TRUTH COVERAGE")
    print(f"  {'-'*40}")
    print(f"  Eval distinct items:  {metrics.get('eval_distinct_items', 0):,}")
    print(f"  Cold-start items:     {metrics.get('cold_start_items', 0):,}")
    for k in [100, 200]:
        gt = metrics.get(f"gt_coverage_at_{k}")
        bs = metrics.get(f"blind_spot_pct_at_{k}")
        if gt is not None:
            print(f"  GT coverage (top-{k}): {gt*100:.1f}%  |  Blind spots: {bs*100:.1f}%")

    # Blind spot miss rates
    bucket_analysis = details.get("bucket_analysis", [])
    if bucket_analysis:
        print(f"\n  MISS RATE BY FREQUENCY (eval-conditioned)")
        print(f"  {'-'*55}")
        print(f"  {'Range':>10} {'Catalog':>8} {'InEval':>8} {'Blind':>8} {'Miss%':>8}")
        for b in bucket_analysis:
            print(
                f"  {b['bucket']:>10} {b['catalog']:>8,} {b['in_eval']:>8,} "
                f"{b['blind']:>8,} {b['miss_rate_eval']*100:>7.1f}%"
            )

    # Repeat vs novel split
    if "repeat_mrr" in metrics:
        _rc = metrics.get("repeat_count", 0)
        _nc = metrics.get("novel_count", 0)
        _rp = metrics.get("repeat_pct", 0) * 100
        _np = metrics.get("novel_pct", 0) * 100
        print(f"\n  REPEAT vs NOVEL SPLIT")
        print(f"  {'-'*60}")
        print(f"  Repeat (target in recent products): {_rc:>10,} ({_rp:.1f}%)")
        print(f"  Novel  (new discovery):              {_nc:>10,} ({_np:.1f}%)")
        print(f"  Recently-viewed baseline MRR:        {metrics['recent_mrr']:>10.4f}")
        print(f"")
        print(f"  {'Metric':<18} {'Repeat':>10} {'Novel':>10} {'Overall':>10}")
        print(f"  {'-'*50}")
        print(f"  {'MRR':<18} {metrics['repeat_mrr']:>10.4f} {metrics['novel_mrr']:>10.4f} {metrics['mrr']:>10.4f}")
        for k in ks:
            rr = metrics.get(f"repeat_recall_at_{k}", 0)
            nr = metrics.get(f"novel_recall_at_{k}", 0)
            ar = metrics.get(f"recall_at_{k}", 0)
            print(f"  {'Recall@' + str(k):<18} {rr:>10.4f} {nr:>10.4f} {ar:>10.4f}")

    print(f"{'='*65}")


def print_masked_novel_summary(metrics):
    """Print the masked novel evaluation results including co-occurrence baseline."""
    ks = sorted(
        int(k.split("_")[-1]) for k in metrics
        if k.startswith("masked_novel_recall_at_")
    )
    if not ks:
        return

    _slices = [
        ("novel_hist", "Has-history"),
        ("novel_first", "First-event"),
        ("novel", "All novel"),
        ("repeat", "Repeat (unmasked)"),
    ]

    has_cat = "masked_novel_cat_pop_mrr" in metrics
    has_cooc = "cooc_novel_hist_mrr" in metrics

    print(f"\n{'='*85}")
    print(f"  MASKED EVALUATION (seen items removed from rankings)")
    print(f"{'='*85}")

    for key, label in _slices:
        prefix = f"masked_{key}"
        cooc_prefix = f"cooc_{key}"
        count = metrics.get(f"{prefix}_count", 0)
        if count == 0:
            continue

        m_mrr = metrics.get(f"{prefix}_mrr", 0)
        p_mrr = metrics.get(f"{prefix}_pop_mrr", 0)
        c_mrr = metrics.get(f"{prefix}_cat_pop_mrr", 0) if has_cat else None
        co_mrr = metrics.get(f"{cooc_prefix}_mrr", None) if has_cooc else None

        print(f"\n  {label} ({count:,} interactions)")

        # Build header dynamically
        hdr = f"  {'Metric':<18} {'Model':>10} {'Global Pop':>12}"
        sep_len = 42
        if has_cat:
            hdr += f" {'Cat Pop':>10}"
            sep_len += 11
        if has_cooc and co_mrr is not None:
            hdr += f" {'Co-occ':>10}"
            sep_len += 11
        print(hdr)
        print(f"  {'-'*sep_len}")

        # MRR row
        row = f"  {'MRR':<18} {m_mrr:>10.4f} {p_mrr:>12.4f}"
        if has_cat:
            row += f" {c_mrr:>10.4f}"
        if has_cooc and co_mrr is not None:
            row += f" {co_mrr:>10.4f}"
        print(row)

        # Recall rows
        for k in ks:
            mr = metrics.get(f"{prefix}_recall_at_{k}", 0)
            pr = metrics.get(f"{prefix}_pop_recall_at_{k}", 0)
            row = f"  {'Recall@' + str(k):<18} {mr:>10.4f} {pr:>12.4f}"
            if has_cat:
                cr = metrics.get(f"{prefix}_cat_pop_recall_at_{k}", 0)
                row += f" {cr:>10.4f}"
            if has_cooc and co_mrr is not None:
                co_r = metrics.get(f"{cooc_prefix}_recall_at_{k}", 0)
                row += f" {co_r:>10.4f}"
            print(row)

    print(f"\n{'='*85}")
