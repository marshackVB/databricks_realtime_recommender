# ---------------------------------------------------------------------------
# Shared feature transforms — single source of truth for training and serving.
#
# Used at training time via mapInPandas and at serving time via the pyfunc
# predict() method (packaged as an MLflow code_paths artifact).
# ---------------------------------------------------------------------------


def transform_item_features(pdf, vocabs=None):
    """Raw Feature View columns → item tower model inputs (pandas)."""
    import numpy as np
    import pandas as pd
    df = pdf.copy()
    df["item_log_price"] = df["item_log_price"].fillna(0.0)
    df["item_popularity"] = np.log1p(df["item_view_count"].fillna(0))
    df["item_conversion_rate"] = (
        (df["item_purchase_count"].fillna(0) + 1) / (df["item_view_count"].fillna(0) + 10)
    )
    df["item_cart_to_purchase_rate"] = (
        (df["item_purchase_count"].fillna(0) + 1) / (df["item_cart_count"].fillna(0) + 10)
    )
    cat1_range = (df["category_l1_max_price"] - df["category_l1_min_price"]).clip(lower=1e-6)
    df["price_pct_rank"] = ((df["item_price"] - df["category_l1_min_price"]) / cat1_range).fillna(0.5)
    cat2_range = (df["category_l2_max_price"] - df["category_l2_min_price"]).clip(lower=1e-6)
    df["price_pct_rank_l2"] = ((df["item_price"] - df["category_l2_min_price"]) / cat2_range).fillna(0.5)
    if "event_time" in df.columns and "item_recency" in df.columns:
        event_dates = pd.to_datetime(df["event_time"]).dt.normalize()
        recency_dates = pd.to_datetime(df["item_recency"]).dt.normalize()
        df["item_recency_val"] = (1.0 / (1.0 + (event_dates - recency_dates).dt.days)).fillna(0.0)
    else:
        df["item_recency_val"] = 0.0
    if vocabs:
        df["item_idx"] = df["product_id"].map(vocabs["item_vocab"]).fillna(0).astype(int)
        df["category_idx"] = df["item_category_l1"].map(vocabs["category_vocab"]).fillna(0).astype(int)
        df["category_l2_idx"] = df["item_category_l2"].map(vocabs["category_l2_vocab"]).fillna(0).astype(int)
        df["category_l3_idx"] = df["item_category_l3"].map(vocabs["category_l3_vocab"]).fillna(0).astype(int)
        df["brand_idx"] = df["item_brand"].map(vocabs["brand_vocab"]).fillna(0).astype(int)
    return df


def transform_user_features(pdf):
    """Raw Feature View columns → user tower model inputs (pandas)."""
    import numpy as np
    import pandas as pd
    import math
    df = pdf.copy()
    for col in ["session_event_count", "session_cart_value", "session_avg_price",
                "session_price_stddev", "session_total_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    # Defensive defaults — RequestSource features are registered and should be
    # passed through by the FE wrapper, but guard against edge cases.
    if "hour_of_day" in df.columns:
        h = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0)
    else:
        h = pd.Series([0.0] * len(df), dtype="float64")
    if "day_of_week" in df.columns:
        d = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0)
    else:
        d = pd.Series([0.0] * len(df), dtype="float64")
    df["hour_sin"] = np.sin(2 * math.pi * h / 24)
    df["hour_cos"] = np.cos(2 * math.pi * h / 24)
    df["dow_sin"] = np.sin(2 * math.pi * d / 7)
    df["dow_cos"] = np.cos(2 * math.pi * d / 7)
    return df


def transform_all_features(pdf, vocabs):
    """Combined transform: item + user. Called from mapInPandas."""
    df = transform_item_features(pdf, vocabs)
    return transform_user_features(df)
