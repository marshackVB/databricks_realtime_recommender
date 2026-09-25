####Project overview:
This project provides an example implementation of a real-time recommender system on Databricks. It implements a two-tower model served in Databricks that performs real-time feature lookups using Databricks Feature Views.

Features are generated on the open source Kaggle dataset, mkechinov/ecommerce-behavior-data-from-multi-category-store, and used to train the model. The dataset contains session-level data over a two-month period.

#####Unity Catalog Assets:
All workspace-specific values (catalog, schema, table names, endpoint names, secret scopes, etc.) are defined in `config.yml`. The keys include:
 - `catalog`, `schema` — UC catalog and schema
 - `ecommerce_table` — clickstream table (backfill source for create_stream(); referenced via StreamSource)
 - `product_table` — product dimension table (referenced via DeltaTableSource)
 - `vs_endpoint`, `vs_index_name` — Vector Search endpoint and index
 - `mlflow_experiment` — MLflow experiment path
 - `online_store` — Lakebase online store name
 - `workspace_host` — workspace URL (used for SP auth and endpoint env vars)
 - `sp_secret_scope` — secret scope with SP client-id and client-secret
 - `kaggle_secret_scope` — secret scope with Kaggle API credentials
 - `run_id_to_deploy` — Stage 1 MLflow run ID to promote to Stage 2

Derived assets (constructed from config at runtime):
 - Embedding table: `{catalog}.{schema}._two_tower_item_embeddings`
 - UC Model: `{catalog}.{schema}.two_tower_user_retrieval`
 - Serving endpoint: `two-tower-recommender` (route-optimized)

##### Directory structure:
All notebooks and Python modules live at the project root (`realtime_recommender/`).
Dev/exploration notebooks live in `realtime_recommender/dev/`.

```
realtime_recommender/
├── AGENT.md
├── config.yml
├── evaluation.py              # Evaluation metrics (build_item_embedding_index, run_full_evaluation, etc.)
├── transforms.py              # Shared transforms — single source of truth for training + serving
├── item_tower_pyfunc.py       # ItemTowerPyfunc class
├── user_tower_pyfunc.py       # UserTowerEmbeddingPyfunc class (Stage 1)
├── user_retrieval_pyfunc.py   # UserTowerRetrievalPyfunc class (Stage 2, embedding + VS)
├── 01_download_datasets.py    # Dataset loading from Kaggle
├── 02_create_features_delta_table_sources.py  # Feature registration + materialization
├── 03_two_tower_train_log.py  # Training + Stage 1 evaluation
├── 04_two_tower_deploy.py     # Stage 2 deployment
└── dev/                       # Dev/exploration notebooks
```

Config is loaded as `os.path.join(os.getcwd(), "config.yml")` — no `..` needed since notebooks are at the root.

##### Feature views.
Both DeltaTableSource and StreamSource Features are calculated on the ecommerce data. DeltaTableSource is used for high-level product features, like product categories, that don't change frequently. StreamSource is used to calculate rolling aggregates on session-level data in near-real time. StreamSource features include features such as the number of items the user has selected during their current session.

RequestSource is used to capture features from within the model serving requests.

Feature Views are materialized to Lakebase tables grouped by entity key:
 - `product_id`: product batch features (ColumnSelection) + product rolling aggregates (event counts, recency)
 - `category_l1`: category-level price bounds (min/max)
 - `category_l2`: category-level price bounds (min/max)
 - `user_session`: session-level rolling aggregates (event count, avg price, stddev, total value)

RequestSource features (product_id, category_l1, category_l2, relevance, session_cart_value, hour_of_day, day_of_week) are passed in the serving request payload and do not persist to Lakebase. They MUST be registered in UC via `fe.register_feature()` (done in `02_create_features_delta_table_sources`) so that `fe.log_model` accepts them in the training_set and the FE serving wrapper passes them through to the pyfunc's `predict()` method.

##### Modeling.
A two-tower Pytorch recommender model is trained. The model will be deployed as a model serving endpoint in Databricks with automatic feature lookups using Databricks Feature Views. The model must be logged to MLflow as a custom Pyfunc model. It will need to authenticate to a Databricks Vector Search Index to retrieve product embeddings. The Pyfunc model will also need to implement any additional feature engineering to convert raw features to model-ready features, though the goal should be to make these transformations simple and fast.

Both product-level and session-level models should be logged to MLflow. An efficient method for generating product-level embeddings and writing them to a Delta table must also be implemented. The Delta table will be synced to a Databricks Vector Index.


##### Special considerations
A valid StreamSource is not available currently. For the time being, DeltaTableSources are used with SlidingWindow, even for within-session features. When StreamSource is available, session-level features must be converted from SlidingWindow to RollingWindow, which is the only window type supported by StreamSource. The aggregation operators (Count, Avg, Sum, StddevPop) are compatible with both window types and will not need to change.

Ultimately, the Feature Engineering Client's create_stream() function will be used to create a stream with an ingestion_destination. In addition, a backfill_source will be provided. All daily session data will be backfilled into the ingestion_destination, with the exception of the final day's worth of data (the last day in November). The backfill table must include a `stream_record_timestamp` column (TIMESTAMP type), which the ingestion pipeline uses to order records and deduplicate against live Kafka events. This last day's session data will be streamed, so that it can populate a Lakebase table to support online-feature joins.

##### MLflow Deployment (Two Tower MLflow Deployment notebook)

**Shared transforms architecture.** Feature transforms (derived numerics, vocab lookups, cyclical encodings, null fills) are defined as plain pandas functions in cell 6 — `transform_item_features(pdf, vocabs)`, `transform_user_features(pdf)`, `transform_all_features(pdf, vocabs)`. These are the single source of truth for both:
  1. Training: applied distributedly via `mapInPandas` with vocabs captured in a closure (serialized via cloudpickle on Spark Connect; `broadcast()` is not available).
  2. Serving: called directly from the pyfunc `predict()` method via the `code_paths` artifact.

`transforms.py` is a standalone file at the project root — the single source of truth for both training (via `mapInPandas`) and serving (via `code_paths` artifact with MLflow). It is NOT auto-extracted via `inspect.getsource()`.

**MLflow code-based logging.** Newer MLflow (>= 3.x) cannot pickle notebook-defined `PythonModel` subclasses. The workaround:
  1. Write each pyfunc class to a standalone `.py` file (e.g., `user_model.py`, `item_model.py`) with `mlflow.models.set_model(...)` at module level.
  2. Import the class from that file before logging (so cloudpickle records a module reference, not `__main__`).
  3. Include the `.py` file in `code_paths` so Spark workers can resolve the module during deserialization.

**code_paths must include ALL referenced modules.** Each `log_model` call needs `code_paths=[transforms_module_path, <model_py_path>]` — both the shared transforms AND the model class file. Without the model `.py` file, workers fail with `ModuleNotFoundError` during `cloudpickle.load`.

**pd.to_numeric coercion in transforms.** Feature View columns arriving from offline/online stores can be `object` dtype (due to nulls in Spark → pandas conversion). The shared transforms must use `pd.to_numeric(df[col], errors="coerce").fillna(0.0)` — not just `.fillna()` — to guarantee float dtype before `torch.tensor()` creation.

**fe.log_model and RequestSource features.** RequestSource features MUST be registered in UC via `fe.register_feature(feature=f, catalog_name=..., schema_name=...)` before they can be included in the `training_set` for `fe.log_model()`. Without registration, `fe.log_model` raises `ValueError: Feature does not have a catalog and schema`. Without inclusion in the training_set, the FE serving wrapper strips RequestSource columns from the DataFrame before calling `predict()`, causing `KeyError` at inference time. The correct pattern (per the `declarative-features-quickstart` example): register all features (including RequestSource) in the `feature_store_integration` notebook, retrieve them via `fe.get_feature()` in the deployment notebook, and include all 25 features in the `log_training_set`. The `transform_user_features` function retains defensive guards for missing columns as a safety net.

##### Serving endpoint deployment.

**Route-optimized endpoints** require `serving_endpoints_data_plane.query()` from the Databricks SDK — NOT `serving_endpoints.query()` or raw HTTP. The data_plane API handles route-optimized URL routing and OAuth authorization_details automatically. Authenticate with the SP client-id/client-secret from the secret scope defined in `config["sp_secret_scope"]`:
```python
import databricks.sdk.core as client
sp_config = client.Config(host=config["workspace_host"], client_id=..., client_secret=...)
sp_w = WorkspaceClient(config=sp_config)
response = sp_w.serving_endpoints_data_plane.query(ENDPOINT_NAME, dataframe_records=[...])
```

**FE-wrapped models do NOT forward `params`/`extra_params` to the inner pyfunc.** The `params` argument in `predict(context, model_input, params=None)` is always `None` when the model is served via `fe.log_model`. To pass runtime configuration (e.g. `top_k`), include it as a column in the input DataFrame and read it from `model_input` inside `predict()`.

**SP permissions for serving:** The SP needs `CAN_QUERY` on the serving endpoint (set in Serving UI), plus UC grants (`USE CATALOG`, `USE SCHEMA`, `SELECT`) on the feature schema for online feature lookups. Grant via the SP's application ID (client-id), not display name.

**fe.score_batch is incompatible with multi-row pyfuncs.** `fe.score_batch` runs the model as a Spark UDF that expects exactly 1 output row per input row. The user tower pyfunc returns top-K recommendations (multiple rows), so `score_batch` raises `OUTPUT_EXCEEDS_INPUT_ROWS`. Validate locally instead: load the traced TorchScript tower, apply shared transforms, generate embedding, query VS directly.

**Delta table CDF for Vector Search.** The Delta table backing a VS delta-sync index must have `delta.enableChangeDataFeed = true`. Set this via `ALTER TABLE ... SET TBLPROPERTIES` after the initial write.

**GPU tensor placement.** When the model is on CUDA, all inputs — including TorchScript tracing examples and batch embedding generation tensors — must be `.to(device)`. Outputs need `.cpu().numpy()` before writing to pandas/Delta.

**mlflow.models.predict() on serverless GPU.** `mlflow.models.predict()` creates an isolated virtualenv subprocess to simulate the serving container. On serverless GPU, this fails because: (1) `virtualenv`/`uv` can't install torch (~2GB) in ephemeral temp; (2) `env_manager="local"` is blocked by Spark Connect; (3) the subprocess lacks CUDA access and Databricks auth. Use direct `mlflow.pyfunc.load_model` + `.predict()` validation instead. Works on classic clusters.

**MLflow pytorch.log_model serialization.** `mlflow.pytorch.log_model` defaults to `pt2` format (requires `input_example`). For two-tower models with complex dict inputs, use `serialization_format="pickle"` or skip full-model logging in favor of per-tower TorchScript traces saved as pyfunc artifacts.