# 📧 Email Triage OpenEnv

A real-world reinforcement learning environment where an AI agent learns to triage emails — classifying urgency, extracting action items, and drafting professional replies.

---

## 🌍 Environment Description

Email triage is a task every professional performs daily. This environment presents an agent with realistic business emails and evaluates its ability to:

1. **Classify urgency** — Is this urgent, normal, or low priority?
2. **Extract action items** — What needs to be done?
3. **Draft a reply** — Write a professional response

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | ID of the current task |
| `task_type` | string | `classify`, `extract`, or `reply` |
| `email_subject` | string | Subject line of the email |
| `email_body` | string | Full body of the email |
| `step` | int | Current step number |
| `done` | bool | Whether episode is complete |

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `action` | string | Free-text response from the agent |

---

## 📋 Tasks

### Task 1 — Email Urgency Classification (Easy)
- **Goal:** Classify email as `urgent`, `normal`, or `low`
- **Grader:** Exact match = 1.0, synonym match = 0.7, wrong = 0.0

### Task 2 — Action Item Extraction (Medium)
- **Goal:** Extract all action items as a comma-separated list
- **Grader:** Partial credit per item found (0.0–1.0)

### Task 3 — Professional Reply Drafting (Hard)
- **Goal:** Draft a complete professional email reply
- **Grader:** Keyword coverage + length bonus (0.0–1.0)

---

## 🚀 Setup & Usage

### Local Development

```bash
pip install -r requirements.txt
uvicorn inference:app --host 0.0.0.0 --port 7860
```

### Test Endpoints

```bash
# Reset
curl -X POST http://localhost:7860/reset

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "urgent"}'

# State
curl http://localhost:7860/state
```

### Run Baseline Agent

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-api-key"

python inference.py --agent
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-3.5-turbo \
  -e HF_TOKEN=your-key \
  email-triage-env
```

---

## 📊 Baseline Scores

| Task | Difficulty | Avg Score |
|---|---|---|
| Email Urgency Classification | Easy | ~0.85 |
| Action Item Extraction | Medium | ~0.65 |
| Professional Reply Drafting | Hard | ~0.55 |

---

## 📁 Project Structure

```
├── env.py           # Environment logic + Pydantic models
├── inference.py     # FastAPI server + baseline agent
├── openenv.yaml     # OpenEnv spec
├── Dockerfile       # Container config
├── requirements.txt
└── README.md
```