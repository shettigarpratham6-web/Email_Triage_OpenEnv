"""
Email Triage OpenEnv — server/app.py
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

# ── Email Dataset ─────────────────────────────────────────────────────────────

EMAILS = [
    {
        "id": "email_001",
        "subject": "URGENT: Production server is down",
        "body": "Our production server has been down for 30 minutes. Customers cannot access the service. Engineers need to be paged immediately.",
        "urgency": "urgent",
        "action_items": ["page engineering team", "notify customers", "check server logs", "escalate to management"],
        "reply_keywords": ["acknowledge", "investigating", "engineering", "update", "apologize"],
    },
    {
        "id": "email_002",
        "subject": "Team lunch next Friday",
        "body": "Hi team, I'd like to organize a lunch next Friday at noon. Please reply if you can make it so I can book a table.",
        "urgency": "low",
        "action_items": ["reply with availability", "confirm attendance"],
        "reply_keywords": ["friday", "attend", "confirm", "lunch", "table"],
    },
    {
        "id": "email_003",
        "subject": "Q3 Report review needed before Monday",
        "body": "Please review the attached Q3 financial report by end of week. Your feedback is required before the stakeholder presentation on Monday.",
        "urgency": "normal",
        "action_items": ["review q3 report", "send feedback", "respond before friday"],
        "reply_keywords": ["review", "feedback", "monday", "report", "stakeholder"],
    },
    {
        "id": "email_004",
        "subject": "CRITICAL: Security breach detected",
        "body": "The security team has detected unauthorized access to the customer database at 3AM. Immediate lockdown and audit are required.",
        "urgency": "urgent",
        "action_items": ["lock down database", "audit access logs", "notify security team", "inform legal"],
        "reply_keywords": ["security", "immediate", "lockdown", "breach", "audit"],
    },
    {
        "id": "email_005",
        "subject": "Monthly newsletter confirmation",
        "body": "Thank you for subscribing to our monthly newsletter. You will receive updates every first Monday of the month. No action needed.",
        "urgency": "low",
        "action_items": ["no action required"],
        "reply_keywords": ["thank", "subscription", "newsletter", "confirm"],
    },
]

TASKS = [
    {"id": "task_easy",   "name": "Email Urgency Classification", "difficulty": "easy",   "type": "classify", "description": "Classify the email urgency as exactly one word: urgent, normal, or low."},
    {"id": "task_medium", "name": "Action Item Extraction",       "difficulty": "medium", "type": "extract",  "description": "List all action items from the email as a comma-separated list."},
    {"id": "task_hard",   "name": "Professional Reply Drafting",  "difficulty": "hard",   "type": "reply",    "description": "Draft a complete professional reply to this email addressing all key points."},
]

# ── Environment ───────────────────────────────────────────────────────────────

class EmailTriageEnv:
    def __init__(self):
        self.task_index = 0
        self.email_index = 0
        self.step_count = 0
        self.done = False
        self.current_task = TASKS[0]
        self.current_email = EMAILS[0]

    def reset(self, seed=None, episode_id=None):
        self.step_count = 0
        self.done = False
        self.current_task = TASKS[self.task_index % len(TASKS)]
        self.current_email = EMAILS[self.email_index % len(EMAILS)]
        return self._make_obs(), {}

    def step(self, action: str):
        self.step_count += 1
        reward, reason = self._grade(action)
        self.done = True
        info = {"task": self.current_task["name"], "difficulty": self.current_task["difficulty"], "reward_reason": reason}
        return self._make_obs(), float(reward), True, False, info

    def state(self):
        return {
            "email_id": self.current_email["id"],
            "task_id": self.current_task["id"],
            "task_type": self.current_task["type"],
            "step_count": self.step_count,
            "done": self.done,
        }

    def set_task(self, task_index: int, email_index: int = 0):
        self.task_index = task_index % len(TASKS)
        self.email_index = email_index % len(EMAILS)

    def _make_obs(self):
        return {
            "task_id": self.current_task["id"],
            "task_type": self.current_task["type"],
            "task_description": self.current_task["description"],
            "email_subject": self.current_email["subject"],
            "email_body": self.current_email["body"],
            "step": self.step_count,
            "done": self.done,
        }

    def _grade(self, action: str):
        task_type = self.current_task["type"]
        a = action.lower().strip()
        if task_type == "classify":
            expected = self.current_email["urgency"]
            if expected in a:
                return 1.0, f"Correct: {expected}"
            synonyms = {"urgent": ["high", "critical", "immediate"], "normal": ["medium", "moderate"], "low": ["minor", "not urgent"]}
            if any(s in a for s in synonyms.get(expected, [])):
                return 0.7, f"Partial: synonym for {expected}"
            return 0.0, f"Wrong: expected {expected}"
        elif task_type == "extract":
            items = self.current_email["action_items"]
            found = sum(1 for item in items if any(w in a for w in item.lower().split()))
            return round(min(1.0, found / len(items)), 2), f"Found {found}/{len(items)} action items"
        elif task_type == "reply":
            keywords = self.current_email["reply_keywords"]
            found = sum(1 for kw in keywords if kw in a)
            score = round(min(1.0, found / len(keywords) + (0.15 if len(action) > 80 else 0)), 2)
            return score, f"Matched {found}/{len(keywords)} key elements"
        return 0.0, "Unknown task type"


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")
env = EmailTriageEnv()


class ResetRequest(BaseModel):
    seed: int | None = None
    episode_id: str | None = None

class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs, info = env.reset(seed=req.seed, episode_id=req.episode_id)
    return {"observation": obs, "info": info}

@app.post("/step")
def step(req: StepRequest):
    action_text = req.action.get("text", req.action.get("action", str(req.action)))
    obs, reward, terminated, truncated, info = env.step(action_text)
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def get_tasks():
    return {"tasks": TASKS}


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()