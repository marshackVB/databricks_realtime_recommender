# ---------------------------------------------------------------------------
# Item tower pyfunc: raw Feature View columns → item embedding vectors.
# Includes transforms.py preprocessing and vocab encoding.
#
# Used in Stage 1 (training evaluation) and Stage 2 (batch embedding generation).
# Logged via mlflow.pyfunc.log_model with code_paths=[transforms.py, this file].
# ---------------------------------------------------------------------------
import mlflow
from mlflow.pyfunc import PythonModel


class ItemTowerPyfunc(PythonModel):
    """Batch pyfunc: raw Feature View columns → item embedding vectors."""

    def load_context(self, context):
        import torch
        import json
        import pickle

        self.device = torch.device("cpu")
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        with open(context.artifacts["vocabs"], "rb") as f:
            self.vocabs = pickle.load(f)
        self.item_tower = torch.jit.load(
            context.artifacts["item_tower"], map_location=self.device
        )
        self.item_tower.eval()

    def predict(self, context, model_input, params=None):
        import torch
        import pandas as pd
        from transforms import transform_item_features

        df = transform_item_features(model_input, self.vocabs)

        with torch.no_grad():
            emb = self.item_tower(
                torch.tensor(df["item_idx"].values, dtype=torch.long),
                torch.tensor(df["category_idx"].values, dtype=torch.long),
                torch.tensor(df["category_l2_idx"].values, dtype=torch.long),
                torch.tensor(df["category_l3_idx"].values, dtype=torch.long),
                torch.tensor(df["brand_idx"].values, dtype=torch.long),
                torch.tensor(df["item_log_price"].values, dtype=torch.float32),
                torch.tensor(df["price_pct_rank"].values, dtype=torch.float32),
                torch.tensor(df["price_pct_rank_l2"].values, dtype=torch.float32),
                torch.tensor(df["item_popularity"].values, dtype=torch.float32),
                torch.tensor(df["item_conversion_rate"].values, dtype=torch.float32),
                torch.tensor(df["item_cart_to_purchase_rate"].values, dtype=torch.float32),
                torch.tensor(df["item_recency_val"].values, dtype=torch.float32),
            )
        emb_np = emb.cpu().numpy()
        return pd.DataFrame(emb_np, columns=[f"emb_{i}" for i in range(emb_np.shape[1])])


mlflow.models.set_model(ItemTowerPyfunc())
