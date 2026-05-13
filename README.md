# Arhis 2026

Reddit-like moderation platform with ML-assisted automatic moderation, built on FastAPI, SQLAlchemy, Alembic, PostgreSQL, Redis, and DVC.

## Frontend

A minimal Svelte frontend lives in [`frontend/`](./frontend/README.md).
It is separate from the backend code and expects the API at `http://localhost:8000` by default.
When started through compose, the frontend is exposed at `http://localhost:8080`.

## What lives in `src`

- `src/app` - API application, DB layer, logging, monitoring, moderation routes
- `src/deploy` - Docker Compose, Prometheus config, Alembic migrations, Dockerfile
- `src/feature_store` - offline and online feature store implementations
- `src/transformations` - text preprocessing and text feature extraction
- `src/pipeline` - DVC stages for preprocessing, training, evaluation, and log materialization

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

## Moderation flow

1. User submits a complaint.
2. The complaint becomes a moderation request.
3. The request goes to the queue.
4. ML worker scores it and returns label + confidence.
5. If confidence is below threshold, the request goes to manual moderation.
6. If confidence is above threshold, the system performs an automatic action.
7. Even high-confidence requests are sampled into manual review for drift detection and retraining data.
8. The final decision and raw logs are written for offline feature-store materialization.

## Logging and monitoring

- Application logs are JSON structured logs on stdout.
- Moderation and comment events are logged with structured fields so they can be replayed into the offline feature store.
- Prometheus metrics are exposed at `GET /metrics`.
- Grafana and Prometheus are wired through Docker Compose.

## Local setup

1. Copy `.env.example` to `.env`
2. Start infrastructure:

```bash
docker compose -f src/deploy/docker-compose.yml up -d postgres redis prometheus grafana
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

## Notes

- Authentication is still a scaffold.
- `users` now stores only identity, auth, and profile data.
- Simple product counters live in `users`; richer ML state stays in the feature-store layer and Redis.
- The model training dataset is currently synthetic on the user side because the public cyberbullying dataset only contains text and labels.
- In production, user-side features should come from the Redis online feature store and from event logs that are periodically materialized into the offline feature store.
