# Model Registry

The registry stores versioned moderation models together with:

- artifact URI
- metadata URI
- feature config version
- validation accuracy
- traffic percent
- lifecycle status

Promotion and rollback are handled by:

- [`src/pipeline/promote_model.py`](../../src/pipeline/promote_model.py)
- [`src/pipeline/rollback_model.py`](../../src/pipeline/rollback_model.py)

See the full lifecycle in [`docs/ml/retraining.md`](./retraining.md).
