# ---------------------------------------------------------------------------
# User tower pyfunc (Stage 1): session features → user embedding.
# No Vector Search retrieval — embedding only, for evaluation and comparison.
#
# Logged via fe.log_model with training_set for feature lineage.
# Stage 2 uses user_retrieval_pyfunc.py which adds VS retrieval on top.
# ---------------------------------------------------------------------------
import mlflow
from mlflow.pyfunc import PythonModel


class UserTowerEmbeddingPyfunc(PythonModel):
    """Stage 1 pyfunc: raw Feature View columns → user embedding (no VS)."""

    def load_context(self, context):
        import torch
        import json

        self.device = torch.device("cpu")
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        self.user_tower = torch.jit.load(
            context.artifacts["user_tower"], map_location=self.device
        )
        self.user_tower.eval()

    def predict(self, context, model_input, params=None):
        import torch
        import pandas as pd
        from transforms import transform_user_features

        df = transform_user_features(model_input)
        numeric_cols = self.config["user_numeric_features"]
        user_numeric = torch.tensor(
            df[numeric_cols].values, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            user_emb = self.user_tower(user_numeric)
        emb_np = user_emb.cpu().numpy()
        return pd.DataFrame(emb_np, columns=[f"emb_{i}" for i in range(emb_np.shape[1])])


mlflow.models.set_model(UserTowerEmbeddingPyfunc())
