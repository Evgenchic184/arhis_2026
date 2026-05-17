# Documentation

This folder is the source of truth for architecture and flow documentation.

## Structure

- [`architecture/c4.md`](./architecture/c4.md) - system context, containers, and backend component view
- [`flows/auth.md`](./flows/auth.md) - registration and login flow
- [`flows/content.md`](./flows/content.md) - posts, comments, replies, and reporting flow
- [`flows/moderation.md`](./flows/moderation.md) - moderator decision flow
- [`flows/admin.md`](./flows/admin.md) - admin role management and feature-config flow
- [`flows/event-pipeline.md`](./flows/event-pipeline.md) - outbox -> Kafka -> parquet/S3 pipeline
- [`ml/retraining.md`](./ml/retraining.md) - retraining triggers, registry lifecycle, canary, and rollback
- [`ml/model-registry.md`](./ml/model-registry.md) - registry overview and storage layout
- [`ops/logging-monitoring.md`](./ops/logging-monitoring.md) - logging, metrics, and operational conventions
- [`runbooks/alerts.md`](./runbooks/alerts.md) - alert response steps
- [`data/event-contracts.md`](./data/event-contracts.md) - event schema and payload conventions

## Reading order

1. Start with [`architecture/c4.md`](./architecture/c4.md).
2. Read the flow docs that match the subsystem you are changing.
3. Use the ops and data docs when wiring workers, sinks, or analytics.
