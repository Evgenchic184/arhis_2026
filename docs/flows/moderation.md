# Moderation Flow

## Report comment

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant FE as Frontend
  participant API as FastAPI API
  participant DB as PostgreSQL
  participant Q as Moderation queue
  participant EVT as Outbox
  participant RELAY as Outbox relay
  participant KAFKA as Kafka ML requests topic

  User->>FE: Report a comment
  FE->>API: POST /api/v1/moderation/comments/{comment_id}/reports
  API->>DB: Validate target comment and reporter
  API->>DB: Insert comment_reports row with pending or queued_for_ml status
  API->>DB: Increment reporter.reports_count
  API->>EVT: Emit comment_report_created
  API->>EVT: Emit moderation_report_routed_to_manual or moderation_report_routed_to_ml
  API->>DB: Commit transaction
  API->>Q: Enqueue report payload for manual queue when needed
  RELAY->>KAFKA: Publish ML request event when routed to ML
  API-->>FE: Report payload
```

## ML inference and escalation

```mermaid
sequenceDiagram
  autonumber
  participant KAFKA as Kafka ML requests topic
  participant ML as ML inference worker
  participant DB as PostgreSQL
  participant Q as Moderation queue
  participant EVT as Outbox

  KAFKA->>ML: Consume report payload
  ML->>ML: Extract features and score comment
  ML->>DB: Update ml_score, ml_verdict, ml_scored_at
  alt confidence below threshold or sampled review
    ML->>DB: Mark report under_review
    ML->>Q: Enqueue report for manual review
    ML->>EVT: Emit moderation_report_escalated_to_manual
  else confidence above threshold
    ML->>DB: Apply auto action
    ML->>EVT: Emit comment_hidden when toxic
    ML->>EVT: Emit moderation_decision_created with decision_source=ml_auto
  end
  ML->>DB: Commit transaction
```

## Moderator decision

```mermaid
sequenceDiagram
  autonumber
  actor Moderator
  participant FE as Frontend
  participant API as FastAPI API
  participant DB as PostgreSQL
  participant EVT as Outbox

  Moderator->>FE: Open moderation queue
  FE->>API: GET /api/v1/moderation/reports
  API->>DB: Read pending reports
  API-->>FE: Report list
  Moderator->>FE: Choose toxic/not_toxic
  FE->>API: POST /api/v1/moderation/reports/{report_id}/decision
  API->>DB: Update report verdict + reviewer fields
  API->>DB: Hide comment when verdict is toxic
  API->>EVT: Emit comment_hidden and moderation_decision_created
  API->>DB: Commit transaction
  API-->>FE: Updated report payload
```

## Notes

- The moderator UI reads the live state from `comment_reports` in Postgres.
- Kafka and the outbox are for downstream processing, ML inference, and replay, not for the live admin screen.
- The ML worker loads versioned artifacts from the model registry and routes 10% of traffic to canary when active.
- The ML worker can auto-resolve high-confidence reports and escalate uncertain or sampled reports to the manual moderation queue.
