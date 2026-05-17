# ADR 0001: Event-backed architecture for moderation and analytics

## Status

Accepted

## Context

The application needs:

- a transactional source of truth for moderation and content state
- an audit trail for user and moderator actions
- an event stream for future ML scoring and offline retraining
- a durable offline history for parquet/S3 materialization

## Decision

We use:

- PostgreSQL for current state and the transactional outbox
- Kafka for replayable domain events
- Parquet in S3 for offline analytics and training data

## Consequences

- The UI stays fast because it reads current state from Postgres.
- Downstream ML and analytics can replay the same events from Kafka.
- The offline feature store can be rebuilt from immutable parquet history.
- The system has one extra moving part, so it needs relay/consumer workers.
