# Authentication Flow

## Register

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant FE as Frontend
  participant API as FastAPI API
  participant DB as PostgreSQL
  participant EVT as Outbox

  User->>FE: Enter username + password
  FE->>API: POST /api/v1/auth/register
  API->>DB: Check username uniqueness
  API->>DB: Create user
  API->>EVT: Emit user_registered
  API->>DB: Commit transaction
  API-->>FE: JWT + user profile
```

## Login

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant FE as Frontend
  participant API as FastAPI API
  participant DB as PostgreSQL
  participant EVT as Outbox

  User->>FE: Enter username + password
  FE->>API: POST /api/v1/auth/login
  API->>DB: Load user
  API->>DB: Update last_login_at
  API->>EVT: Emit user_logged_in
  API->>DB: Commit transaction
  API-->>FE: JWT + user profile
```

## Notes

- The first registered account becomes `admin`.
- All subsequent accounts default to `user`.
- The frontend stores the JWT in local storage and sends `Authorization: Bearer <token>`.

