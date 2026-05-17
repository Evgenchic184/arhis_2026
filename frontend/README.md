# Arhis Frontend

Minimal Svelte UI for the Arhis moderation app.

## Run

```bash
cd frontend
npm install
npm run dev
```

By default the frontend auto-detects the backend as `<current-host>:8000`, so it works both on `localhost` and on your laptop's LAN IP. Set `VITE_API_BASE_URL` in `.env` only if you need a custom backend host.

## Docker

The frontend can also be built and served as a separate container through the root `docker-compose.yml`.
It is exposed on `http://localhost:8080` in the default compose setup, and also works when opened from another device via the laptop's LAN IP.

## Features

- JWT login and registration
- post feed and post creation
- post detail with comments and replies
- comment reporting
- moderator queue with verdict actions
- admin user management with role changes
- admin ML registry view with promote / rollback controls
- mobile and desktop responsive layout
