# Event Pipeline

## Outbox to Kafka to parquet

```mermaid
sequenceDiagram
  autonumber
  participant API as FastAPI API
  participant DB as PostgreSQL
  participant OUTBOX as domain_event_outbox
  participant RELAY as Outbox relay worker
  participant KAFKA as Kafka
  participant SINK as Parquet sink worker
  participant S3 as S3 / parquet storage

  API->>DB: Write business state + outbox row in one transaction
  DB-->>OUTBOX: Persist domain event
  RELAY->>DB: Read unpublished outbox rows
  RELAY->>KAFKA: Publish event record
  RELAY->>DB: Mark row published_at
  SINK->>KAFKA: Consume domain events and ML request events
  SINK->>S3: Write partitioned parquet file
  SINK->>KAFKA: Commit offsets after successful upload
```

## Moderation ML request path

```mermaid
sequenceDiagram
  autonumber
  participant API as FastAPI API
  participant DB as PostgreSQL
  participant OUTBOX as domain_event_outbox
  participant RELAY as Outbox relay worker
  participant KAFKA as Kafka moderation ML topic
  participant ML as ML inference worker

  API->>DB: Write report state and outbox event in one transaction
  DB-->>OUTBOX: Persist moderation_report_routed_to_ml
  RELAY->>DB: Read unpublished outbox rows
  RELAY->>KAFKA: Publish ML request event
  ML->>KAFKA: Consume request event
  ML->>DB: Save prediction, confidence, and decision source
```

## Model registry deployment

```mermaid
sequenceDiagram
  autonumber
  participant TRAIN as Training pipeline
  participant VAL as Validation sample
  participant DB as PostgreSQL registry
  participant S3 as MinIO / S3
  participant ML as ML inference worker

  TRAIN->>VAL: Score a small validation sample
  VAL-->>TRAIN: Return accuracy and quality metrics
  alt accuracy == 100%
    TRAIN->>S3: Upload model artifact and metadata
    TRAIN->>DB: Insert model_versions row
    TRAIN->>DB: Mark model as production or canary
    ML->>DB: Read active production/canary versions
  else accuracy < 100%
    TRAIN->>DB: Insert rejected registry row
  end
```

## Retraining and deployment control

```mermaid
sequenceDiagram
  autonumber
  participant MON as retraining-monitor
  participant S3 as MinIO parquet archive
  participant REG as model registry
  participant TR as retrain_model
  participant PROMOTE as promote_model
  participant ROLLBACK as rollback_model

  MON->>S3: Read recent parquet events
  MON->>REG: Load active production snapshot
  MON->>MON: Compute PSI, token share, sampled accuracy
  alt trigger fired
    MON->>TR: Launch retraining job
    TR->>REG: Register validated canary version
  else no trigger
    MON->>MON: Keep monitoring
  end
  alt canary health is good
    PROMOTE->>REG: Promote canary to production
  else health degrades
    ROLLBACK->>REG: Roll back production to previous version
  end
```

## Design rules

- Business state is written to Postgres first.
- Domain events are persisted in the outbox in the same transaction.
- Kafka is used for replayable event streaming.
- Moderation reports can also be routed to a dedicated ML request topic.
- Parquet is the offline, append-only sink.
- Object storage is the long-term archive for offline analytics and ML retraining.
- The parquet sink commits Kafka offsets only after the object upload succeeds.
- Model registry deployment is blocked unless the sampled validation accuracy reaches 100%.
- Retraining monitor is the drift gate that decides when to launch a new training run.
- Promotion and rollback are explicit registry actions, not automatic side effects of training.
