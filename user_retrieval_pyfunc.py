# ---------------------------------------------------------------------------
# User tower retrieval pyfunc (Stage 2): session features → user embedding
# → Vector Search → top-K product recommendations.
#
# Logged via fe.log_model with training_set for feature lineage.
# Deployed to a model serving endpoint with automatic feature lookup.
# ---------------------------------------------------------------------------
import mlflow
from mlflow.pyfunc import PythonModel


class UserTowerRetrievalPyfunc(PythonModel):
    """Stage 2 serving pyfunc: raw features → user embedding → VS → top-K."""

    def load_context(self, context):
        import torch
        import json
        import os
        from databricks.vector_search.client import VectorSearchClient

        self.device = torch.device("cpu")
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        self.user_tower = torch.jit.load(
            context.artifacts["user_tower"], map_location=self.device
        )
        self.user_tower.eval()
        self.vsc = VectorSearchClient(
            workspace_url=os.environ["DATABRICKS_HOST"],
            service_principal_client_id=os.environ["DATABRICKS_CLIENT_ID"],
            service_principal_client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
        )
        self.vs_endpoint = self.config["vs_endpoint"]
        self.vs_index = (
            f"{self.config['catalog']}.{self.config['schema']}"
            f".{self.config['vs_index_name']}"
        )

    def predict(self, context, model_input, params=None):
        import torch
        import pandas as pd
        from transforms import transform_user_features

        df = transform_user_features(model_input)
        top_k = 10
        if params and "top_k" in params:
            top_k = int(params["top_k"])


        numeric_cols = self.config["user_numeric_features"]
        user_numeric = torch.tensor(
            df[numeric_cols].values, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            user_emb = self.user_tower(user_numeric)
        emb_list = user_emb[0].cpu().numpy().tolist()

        try:
            idx = self.vsc.get_index(
                endpoint_name=self.vs_endpoint, index_name=self.vs_index
            )
            results = idx.similarity_search(
                query_vector=emb_list,
                columns=["product_id", "item_category_l1", "item_brand", "item_price"],
                num_results=top_k,
            )
            result_df = pd.DataFrame(
                results.get("result", {}).get("data_array", []),
                columns=[
                    c["name"]
                    for c in results.get("manifest", {}).get("columns", [])
                ],
            )
            return result_df
        except Exception as e:
            return pd.DataFrame({"user_embedding": [emb_list], "error": [str(e)]})


mlflow.models.set_model(UserTowerRetrievalPyfunc())
