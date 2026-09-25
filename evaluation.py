# ---------------------------------------------------------------------------
# Evaluation metrics for the two-tower recommender model.
#
# Adapted from two_tower_ecommerce_aggregates_only (cells 9-14).
# Provides reusable functions for Recall@K, MRR, NDCG@K, popularity
# baseline, catalog coverage, blind spot analysis, and embedding geometry.
#
# Usage:
#   from evaluation import run_full_evaluation, generate_evaluation_figures
#   metrics, details = run_full_evaluation(model, train_pdf, eval_pdf, eval_loader, device)
#   figures = generate_evaluation_figures(metrics, details)
# ---------------------------------------------------------------------------
import torch
import numpy as np
import math

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


# ---- Core evaluation functions ----


def build_item_embedding_index(model, train_pdf, device, batch_size=1024):
    """Build L2-normalized item embeddings for all unique training items.

    Returns:
        item_embs: (num_items, embed_dim) tensor on device
        item_idx_list: numpy array mapping row position to item_idx
        item_features: deduplicated DataFrame (one row per item)
    """
    model.eval()
    cols = list(ITEM_EMB_COLUMNS.keys())
    item_features = (
        train_pdf.drop_duplicates("item_idx")[cols]
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

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


def evaluate_retrieval_metrics(model, eval_loader, item_embs, item_idx_list,
                               device, ks=(10, 50, 100, 150, 200)):
    """Compute Recall@K, MRR, and NDCG@K on the eval set."""
    model.eval()
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0

    max_item_idx = int(item_idx_list.max()) + 1
    idx_to_pos = torch.full((max_item_idx,), -1, dtype=torch.long, device=device)
    for pos, idx in enumerate(item_idx_list):
        idx_to_pos[int(idx)] = pos

    item_idx_tensor = torch.tensor(item_idx_list)

    with torch.no_grad():
        for user_batch, item_batch in eval_loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            user_emb = model.user_tower(**user_batch)
            true_item_idx = item_batch["item_id"]

            scores = user_emb @ item_embs.T
            topk_positions = scores.topk(max_k, dim=1).indices.cpu()
            topk_item_ids = item_idx_tensor[topk_positions]
            match_mask = topk_item_ids == true_item_idx.unsqueeze(1)

            for k in ks:
                hits[k] += match_mask[:, :k].any(dim=1).sum().item()
                rank_in_k = match_mask[:, :k].float().argmax(dim=1)
                hit_in_k = match_mask[:, :k].any(dim=1).float()
                dcg = hit_in_k / torch.log2(rank_in_k.float() + 2)
                ndcg_sums[k] += dcg.sum().item()

            # MRR (fully vectorized on GPU)
            true_gpu = true_item_idx.to(device).clamp(0, max_item_idx - 1)
            true_pos = idx_to_pos[true_gpu]
            valid = true_pos >= 0
            if valid.any():
                true_scores = scores[torch.arange(scores.size(0), device=device), true_pos]
                ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1).float() + 1
                rr_sum += (valid.float() / ranks).sum().item()

            total += len(true_item_idx)

    catalog_size = len(item_idx_list)
    metrics = {"mrr": rr_sum / total, "eval_interactions": total, "catalog_size": catalog_size}
    for k in ks:
        metrics[f"recall_at_{k}"] = hits[k] / total
        metrics[f"ndcg_at_{k}"] = ndcg_sums[k] / total
        metrics[f"random_recall_at_{k}"] = min(k / catalog_size, 1.0)
    return metrics, total


def compute_popularity_baseline(train_pdf, eval_loader, ks=(10, 50, 100, 150, 200)):
    """Popularity baseline: always recommend the most-interacted items."""
    pop_counts = train_pdf["item_idx"].value_counts()
    pop_ranked = pop_counts.index.values
    pop_rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(pop_ranked)}
    default_rank = len(pop_ranked) + 1

    pop_hits = {k: 0 for k in ks}
    pop_rr_sum = 0.0
    pop_ndcg_sums = {k: 0.0 for k in ks}
    pop_total = 0

    for _, item_batch in eval_loader:
        for true_idx in item_batch["item_id"].numpy():
            rank = pop_rank_lookup.get(int(true_idx), default_rank)
            pop_rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    pop_hits[k] += 1
                    pop_ndcg_sums[k] += 1.0 / math.log2(rank + 1)
        pop_total += len(item_batch["item_id"])

    metrics = {"pop_mrr": pop_rr_sum / pop_total}
    for k in ks:
        metrics[f"pop_recall_at_{k}"] = pop_hits[k] / pop_total
        metrics[f"pop_ndcg_at_{k}"] = pop_ndcg_sums[k] / pop_total
    return metrics, pop_ranked


def compute_catalog_coverage(model, eval_loader, item_embs, item_idx_list,
                             device, ks=(10, 50, 100, 200)):
    """Fraction of catalog surfaced in any session's top-K candidates."""
    model.eval()
    max_k = max(ks)
    retrieved_sets = {k: set() for k in ks}
    item_idx_tensor = torch.tensor(item_idx_list)

    with torch.no_grad():
        for user_batch, _ in eval_loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            user_emb = model.user_tower(**user_batch)
            scores = user_emb @ item_embs.T
            topk_positions = scores.topk(max_k, dim=1).indices.cpu()
            topk_item_ids = item_idx_tensor[topk_positions]
            for k in ks:
                retrieved_sets[k].update(topk_item_ids[:, :k].reshape(-1).tolist())

    catalog_size = len(item_idx_list)
    metrics = {}
    for k in ks:
        n = len(retrieved_sets[k])
        metrics[f"coverage_at_{k}"] = n / catalog_size
        metrics[f"unique_retrieved_at_{k}"] = n
    return metrics, retrieved_sets


def analyze_retrieval_vs_ground_truth(eval_pdf, retrieved_sets, item_idx_list,
                                      ks=(100, 200)):
    """Coverage vs eval ground truth and blind spot identification."""
    eval_true = set(eval_pdf["item_idx"].unique().tolist())
    train_catalog = set(item_idx_list.tolist())
    cold_start = eval_true - train_catalog

    metrics = {
        "eval_distinct_items": len(eval_true),
        "cold_start_items": len(cold_start),
        "cold_start_pct": len(cold_start) / max(len(eval_true), 1),
    }
    blind_spots, warm_blind = {}, {}
    for k in ks:
        covered = retrieved_sets[k] & eval_true
        blind_k = eval_true - retrieved_sets[k]
        warm_k = blind_k - cold_start
        metrics[f"gt_coverage_at_{k}"] = len(covered) / max(len(eval_true), 1)
        metrics[f"blind_spots_at_{k}"] = len(blind_k)
        metrics[f"blind_spot_pct_at_{k}"] = len(blind_k) / max(len(eval_true), 1)
        metrics[f"warm_blind_at_{k}"] = len(warm_k)
        blind_spots[k] = blind_k
        warm_blind[k] = warm_k

    return metrics, blind_spots, warm_blind, cold_start, train_catalog


def analyze_blind_spots(train_pdf, blind_spot_ids, retrieved_ids, train_catalog,
                        eval_pdf):
    """Blind spot distribution by training interaction count.

    Reports both catalog-basis and eval-conditioned miss rates to avoid
    denominator artifacts (per evaluation methodology: the covariate
    'training frequency' also drives eval-set eligibility).
    """
    all_counts = train_pdf["item_idx"].value_counts()
    bs_counts = train_pdf[train_pdf["item_idx"].isin(blind_spot_ids)]["item_idx"].value_counts()
    ret_counts = train_pdf[train_pdf["item_idx"].isin(retrieved_ids)]["item_idx"].value_counts()
    eval_gt = set(eval_pdf["item_idx"].unique())

    metrics = {
        "bs_median_interactions": float(bs_counts.median()) if len(bs_counts) else 0,
        "bs_mean_interactions": float(bs_counts.mean()) if len(bs_counts) else 0,
        "ret_median_interactions": float(ret_counts.median()) if len(ret_counts) else 0,
        "ret_mean_interactions": float(ret_counts.mean()) if len(ret_counts) else 0,
    }

    bucket_analysis = []
    for lo, hi in INTERACTION_BUCKETS:
        label = _bucket_label(lo, hi)
        cat_ids = _bucket_ids(all_counts, lo, hi)
        cat_n = len(cat_ids)
        eval_ids = cat_ids & eval_gt
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


def run_full_evaluation(model, train_pdf, eval_pdf, eval_loader, device,
                        ks=(10, 50, 100, 150, 200), coverage_ks=(10, 50, 100, 200)):
    """Run all evaluation metrics.

    Returns:
        metrics: flat dict of all numeric metrics (ready for mlflow.log_metrics)
        details: dict of detailed objects for downstream use / artifact logging
    """
    print("Building item embedding index...")
    item_embs, item_idx_list, item_features = build_item_embedding_index(
        model, train_pdf, device
    )

    print("Computing Recall@K, MRR, NDCG@K...")
    retrieval, total = evaluate_retrieval_metrics(
        model, eval_loader, item_embs, item_idx_list, device, ks
    )

    print("Computing popularity baseline...")
    pop, pop_ranked = compute_popularity_baseline(train_pdf, eval_loader, ks)

    # Lift over popularity
    lifts = {}
    for k in ks:
        lifts[f"recall_lift_at_{k}"] = retrieval[f"recall_at_{k}"] / max(pop[f"pop_recall_at_{k}"], 1e-10)
        lifts[f"ndcg_lift_at_{k}"] = retrieval[f"ndcg_at_{k}"] / max(pop[f"pop_ndcg_at_{k}"], 1e-10)
    lifts["mrr_lift"] = retrieval["mrr"] / max(pop["pop_mrr"], 1e-10)

    print("Computing catalog coverage...")
    cov, retrieved_sets = compute_catalog_coverage(
        model, eval_loader, item_embs, item_idx_list, device, coverage_ks
    )

    # Popularity concentration in top-100
    if 100 in retrieved_sets:
        top1k = set(pop_ranked[:1000].tolist())
        overlap = retrieved_sets[100] & top1k
        cov["pop_concentration_100"] = len(overlap) / max(len(retrieved_sets[100]), 1)

    print("Analyzing retrieval vs ground truth...")
    gt, blind_spots, warm_blind, cold_start, train_catalog = (
        analyze_retrieval_vs_ground_truth(eval_pdf, retrieved_sets, item_idx_list)
    )

    print("Analyzing blind spots...")
    blind_200 = warm_blind.get(200, set())
    retrieved_200 = retrieved_sets.get(200, set()) & train_catalog
    bs, bucket_analysis = analyze_blind_spots(
        train_pdf, blind_200, retrieved_200, train_catalog, eval_pdf
    )

    print("Analyzing embedding geometry...")
    all_counts = train_pdf["item_idx"].value_counts()
    emb, bucket_results = analyze_embedding_geometry(
        item_embs.cpu().numpy(), item_idx_list, item_features,
        all_counts, blind_200
    )

    # Merge all metrics
    metrics = {}
    for d in [retrieval, pop, lifts, cov, gt, bs, emb]:
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

    print(f"{'='*65}")
