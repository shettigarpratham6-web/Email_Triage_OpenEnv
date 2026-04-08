from pydantic import BaseModel
from typing import Any

# ── Typed Models (OpenEnv spec) ──────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_type: str          # classify | extract | reply
    email_subject: str
    email_body: str
    step: int
    done: bool

class Action(BaseModel):
    action: str

class Reward(BaseModel):
    value: float
    reason: str

# ── Email Dataset ────────────────────────────────────────────────────────────

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
        "action_items": ["lock down database", "audit access logs", "notify security team", "inform legal", "alert customers"],
        "reply_keywords": ["security", "immediate", "lockdown", "breach", "audit", "team"],
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
    {
        "id": "task_easy",
        "name": "Email Urgency Classification",
        "difficulty": "easy",
        "description": "Classify the email urgency. Reply with exactly one word: urgent, normal, or low.",
        "type": "classify",
    },
    {
        "id": "task_medium",
        "name": "Action Item Extraction",
        "difficulty": "medium",
        "description": "List all action items from the email as a comma-separated list.",
        "type": "extract",
    },
    {
        "id": "task_hard",
        "name": "Professional Reply Drafting",
        "difficulty": "hard",
        "description": "Draft a complete professional reply to this email addressing all key points.",
        "type": "reply",
    },
]

# ── Environment ──────────────────────────────────────────────────────────────

class EmailTriageEnv:
    def __init__(self):
        self.task_index = 0
        self.email_index = 0
        self.step_count = 0
        self.done = False
        self.current_task = TASKS[0]
        self.current_email = EMAILS[0]

    # ── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self):
        self.step_count = 0
        self.done = False
        self.current_task = TASKS[self.task_index % len(TASKS)]
        self.current_email = EMAILS[self.email_index % len(EMAILS)]
        obs = self._make_obs()
        return obs, {}

    def step(self, action: str):
        self.step_count += 1
        reward_value, reason = self._grade(action)
        self.done = True

        obs = self._make_obs()
        info = {
            "task": self.current_task["name"],
            "difficulty": self.current_task["difficulty"],
            "reward_reason": reason,
        }
        return obs, float(reward_value), True, False, info

    def state(self):
        return {
            "email_id": self.current_email["id"],
            "task_id": self.current_task["id"],
            "task_type": self.current_task["type"],
            "step_count": self.step_count,
            "done": self.done,
            "task_index": self.task_index,
            "email_index": self.email_index,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_obs(self) -> dict:
        return Observation(
            task_id=self.current_task["id"],
            task_type=self.current_task["type"],
            email_subject=self.current_email["subject"],
            email_body=self.current_email["body"],
            step=self.step_count,
            done=self.done,
        ).dict()

    def _grade(self, action: str):
        task_type = self.current_task["type"]
        a = action.lower().strip()

        if task_type == "classify":
            expected = self.current_email["urgency"]
            if expected in a:
                return 1.0, f"Correct: {expected}"
            # Partial credit for synonyms
            synonyms = {
                "urgent": ["high", "critical", "immediate", "asap"],
                "normal": ["medium", "moderate", "regular"],
                "low": ["minor", "not urgent", "whenever", "low priority"],
            }
            if any(s in a for s in synonyms.get(expected, [])):
                return 0.7, f"Partial: synonym for {expected}"
            return 0.0, f"Wrong: expected {expected}"

        elif task_type == "extract":
            items = self.current_email["action_items"]
            found = sum(
                1 for item in items
                if any(word in a for word in item.lower().split())
            )
            score = round(min(1.0, found / len(items)), 2)
            return score, f"Found {found}/{len(items)} action items"

        elif task_type == "reply":
            keywords = self.current_email["reply_keywords"]
            found = sum(1 for kw in keywords if kw in a)
            base = found / len(keywords)
            length_bonus = 0.15 if len(action) > 80 else 0
            score = round(min(1.0, base + length_bonus), 2)
            return score, f"Matched {found}/{len(keywords)} key elements"

        return 0.0, "Unknown task type"

    def set_task(self, task_index: int, email_index: int = 0):
        self.task_index = task_index % len(TASKS)
        self.email_index = email_index % len(EMAILS)