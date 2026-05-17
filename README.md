# Arhis 2026

Reddit-like moderation platform with ML-assisted automatic moderation, built on FastAPI, SQLAlchemy, Alembic, PostgreSQL, Redis, and DVC.

## Docs

Detailed architecture and flow documentation lives in [`docs/`](./docs/README.md).

## Frontend

A minimal Svelte frontend lives in [`frontend/`](./frontend/README.md).
It is separate from the backend code and expects the API at `http://localhost:8000` by default.
When started through compose, the frontend is exposed at `http://localhost:8080`.

## What lives in `src`

- `src/app` - API application, DB layer, logging, monitoring, moderation routes
- `src/deploy` - Docker Compose, Prometheus config, Alembic migrations, Dockerfile
- `src/feature_store` - offline and online feature store implementations
- `src/transformations` - text preprocessing and text feature extraction
- `src/pipeline` - DVC stages for preprocessing, training, evaluation, log materialization, and retraining
- `src/app/workers` - outbox relay, ML inference, parquet sink, and retraining monitor workers
- `src/app/services` - moderation routing, model registry, and supporting services

## Data sources

- Raw dataset: `data/cyberbullying_tweets.csv`
- Columns in the source data:
  - `tweet_text`
  - `cyberbullying_type`
- Existing artifacts in `data/`:
  - `all_data.parquet`
  - `train.parquet`
  - `val.parquet`
  - `test.parquet`
  - `with_user_features.parquet`

## Feature set

### Base features used by the current training pipeline

These are the exact columns present in `train.parquet`, `val.parquet`, and `test.parquet` and used by the model:

- `tweet_text`
- `cyberbullying_type`
- `cyberbullying_bin`
- `text_prepared`
- `text_length`
- `caps_ratio`
- `has_url`
- `has_mention`
- `user_id`
- `is_new_user`
- `reputation_score`
- `reports_last_24h`
- `account_age_days`

Important:

- `cyberbullying_bin` is the target label, not an input feature.
- `cyberbullying_type` is source metadata for analysis and evaluation, not an input feature at inference time.
- The model only consumes the feature subset defined in `src/feature_store/feature_sets.py`.

### Extension features kept for future iterations

The codebase also keeps helper functions and schemas for richer text/user features such as:

- `token_count`
- `alpha_ratio`
- `punctuation_ratio`
- `digit_ratio`
- `avg_token_length`
- `unique_token_ratio`
- `num_exclamation_marks`
- `num_question_marks`
- `num_digits`
- `repeated_char_sequences`
- `toxic_keyword_hits`
- `reports_last_7d`
- `reports_last_30d`
- `deleted_comments_last_1d`
- `deleted_comments_last_7d`
- `deleted_comments_last_30d`
- `hidden_comments_last_1d`
- `hidden_comments_last_7d`
- `hidden_comments_last_30d`
- `comment_count_last_1d`
- `comment_count_last_7d`
- `comment_count_last_30d`
- `auto_action_count_last_30d`
- `manual_overrule_count_last_30d`
- `auto_action_rate_last_30d`
- `manual_overrule_rate_last_30d`
- `last_ml_confidence`
- `last_ml_verdict`

## Runtime feature config

Feature selection is configurable at runtime:

- `params.yaml` seeds the default runtime feature config under `features.runtime`
- `GET /api/v1/config/features` returns the active config and the allowed feature lists
- `PATCH /api/v1/config/features` updates the active config in Redis-backed storage

Current rules:

- `training_feature_columns` controls which model features are used during training
- `inference_feature_columns` controls which model features are used during inference
- `online_user_feature_columns` controls which user features are read from the online feature store
- `text_column` is currently fixed to `text_prepared`
- `last_ml_verdict` is stored as an expandable online/offline feature, but is not part of the current model inputs because the pipeline does not encode categorical features yet

## Model registry and canary

- Models are stored as versioned artifacts in MinIO/S3 under `MODEL_REGISTRY_BUCKET_NAME`.
- Registry state is tracked in Postgres in `model_versions`.
- The first validated model becomes `production`.
- Later validated models become `canary` and receive 10% of inference traffic by default.
- New candidates must pass the validation gate on a small sample with 100% accuracy, otherwise the deploy is rejected.
- The ML worker routes traffic deterministically by `report_id` so the canary split is stable.

### Canary test pipeline

For testing canary auto-promote / rollback, there is a separate intentionally weaker pipeline:

- `uv run python -m src.pipeline.train_canary_model`
- `uv run python -m src.pipeline.register_canary_model`

It trains on 10% of `data/train.parquet` and adds controlled label noise so the canary model is usually worse than the normal model.
- `uv run python -m src.pipeline.register_model` registers the trained artifact, uploads it to object storage, and writes the registry record.
- Retraining datasets are stored in a separate MinIO/S3 bucket under `RETRAINING_DATASET_BUCKET_NAME` as versioned parquet bundles with train/val/test splits, summaries, manifests, and baseline vocabulary snapshots.

## Retraining and drift monitoring

- `retraining-monitor` reads parquet archives from MinIO, computes PSI, new-token share, sampled manual accuracy, and data freshness.
- When retraining triggers fire, it launches `uv run python -m src.pipeline.retrain_model`.
- `src/pipeline/build_retraining_dataset.py` builds a training dataset from event parquet logs.
- `src/pipeline/retrain_model.py` trains a candidate, validates it on a small sample, writes it to the registry, and archives the retraining dataset bundle into the retraining dataset bucket.
- Prometheus exposes the retraining and model-monitoring metrics through the worker metrics endpoints.
- Grafana dashboard panels include PSI, token novelty, sampled accuracy, freshness, canary share, and registry events.

## Moderation flow

1. User submits a complaint.
2. The complaint becomes a moderation request.
3. The request is written to the transactional `comment_reports` table.
4. The same action is also written to the `domain_event_outbox` table in the same DB transaction.
5. Reports are split at write time:
   - some go directly to the manual moderation queue
   - some are routed to the ML request topic
6. A relay worker publishes outbox events to Kafka.
7. The ML inference worker consumes the ML request topic, scores the report, and decides between auto action and manual escalation.
8. A downstream consumer can write Kafka events to parquet files for offline replay, analytics, and future ML training.
9. The moderator UI reads current queue state from Postgres, not from Kafka.

## Logging and monitoring

- Application logs are JSON structured logs on stdout.
- Domain events are also persisted in the Postgres outbox, then relayed to Kafka for durable replay.
- Kafka consumers can materialize parquet snapshots for offline feature storage and later sync them to MinIO/S3.
- Prometheus metrics are exposed at `GET /metrics`.
- Grafana and Prometheus are wired through Docker Compose.
- Grafana auto-loads the `Arhis Business Metrics` dashboard from provisioning.
- Kafka UI is exposed on `http://localhost:8081`.
- MinIO console is exposed on `http://localhost:9001`.
- The ML inference worker consumes `arhis.moderation.ml.requests`.
- The ML inference worker metrics endpoint is exposed on `http://localhost:8011/metrics`.
- The retraining monitor metrics endpoint is exposed on `http://localhost:8012/metrics`.

Optional workers:

```bash
uv run python -m src.app.workers.outbox_relay
uv run python -m src.app.workers.ml_inference_worker
uv run python -m src.app.workers.parquet_sink
uv run python -m src.app.workers.retraining_monitor
```

Set `KAFKA_BOOTSTRAP_SERVERS` and the `S3_*` variables to point them at your broker and object store.

## Local setup

1. Copy `.env.example` to `.env`
2. Start infrastructure:

```bash
docker compose -f src/deploy/docker-compose.yml up -d
```

3. Run migrations:

```bash
uv run alembic upgrade head
```

4. Start the API:

```bash
uv run uvicorn src.app.main:app --reload
```

## DVC pipeline

- Preprocess raw data:

```bash
uv run python -m src.pipeline.preprocess_data
```

- Materialize offline feature store from logs:

```bash
uv run python -m src.pipeline.materialize_offline_feature_store
```

- Train model:

```bash
uv run python -m src.pipeline.train_model
```

- Evaluate model:

```bash
uv run python -m src.pipeline.evaluate_model
```

- Register model into registry / canary:

```bash
uv run python -m src.pipeline.register_model
```

- Build retraining dataset from parquet logs:

```bash
uv run python -m src.pipeline.build_retraining_dataset
```

- Retrain, validate, and register a new model version:

```bash
uv run python -m src.pipeline.retrain_model
```

- Promote a canary version to production:

```bash
uv run python -m src.pipeline.promote_model
```

- Roll back production to the previous version:

```bash
uv run python -m src.pipeline.rollback_model
```

- Reproduce the full pipeline:

```bash
uv run dvc repro
```

## API hints

- Authentication now uses JWT.
- Register with `username + password`; the first registered account becomes `admin`.
- Use `Authorization: Bearer <access_token>` for protected endpoints.
- Admins can promote users to `moderator` or `admin` through `PATCH /api/v1/users/{user_id}/role`.

## Main endpoints

- `GET /health`
- `GET /metrics`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/posts`
- `GET /api/v1/posts`
- `PATCH /api/v1/posts/{post_id}`
- `DELETE /api/v1/posts/{post_id}`
- `POST /api/v1/posts/{post_id}/comments`
- `PATCH /api/v1/comments/{comment_id}`
- `DELETE /api/v1/comments/{comment_id}`
- `POST /api/v1/moderation/comments/{comment_id}/reports`
- `GET /api/v1/moderation/reports`
- `POST /api/v1/moderation/reports/{report_id}/decision`
- `GET /api/v1/config/features`
- `PATCH /api/v1/config/features`
- `GET /api/v1/users`
- `PATCH /api/v1/users/{user_id}/role`
- `GET /api/v1/models`
- `POST /api/v1/models/{version}/promote`
- `POST /api/v1/models/rollback`

## Notes

- Authentication uses JWT.
- `users` now stores only identity, auth, and profile data.
- Simple product counters live in `users`; richer ML state stays in the feature-store layer and Redis.
- The model training dataset is currently synthetic on the user side because the public cyberbullying dataset only contains text and labels.
- In production, user-side features should come from the Redis online feature store and from event logs that are periodically materialized into the offline feature store.
- The default local event pipeline is `Postgres outbox -> Kafka -> parquet -> MinIO`.
- Prometheus business metrics include:
  - `arhis_users_total`
  - `arhis_posts_created_total`
  - `arhis_comments_created_total`
  - `arhis_reports_created_total`
  - `arhis_comments_hidden_total`
  - `arhis_moderation_decision_latency_seconds`
- The Grafana dashboard uses `increase(...[5m])` for the last-5-minute counters and `rate(sum)/rate(count)` for average moderation latency.
