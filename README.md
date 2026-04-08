---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📧 Email Triage OpenEnv

A real-world RL environment where an AI agent triages emails — classifying urgency, extracting action items, and drafting professional replies.

## Endpoints
- `POST /reset` — Start new episode
- `POST /step` — Take action `{"action": {"text": "your response"}}`
- `GET /state` — Current environment state
- `GET /health` — Health check
- `GET /docs` — Swagger UI

## Tasks
1. **Easy** — Urgency Classification (`urgent` / `normal` / `low`)
2. **Medium** — Action Item Extraction
3. **Hard** — Professional Reply Drafting

## Setup
```bash
uv sync
uv run server
```