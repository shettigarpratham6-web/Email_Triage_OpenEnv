"""
Email Triage OpenEnv — inference.py
Serves the OpenEnv API and (when run directly) executes the baseline agent.
"""

import os
import sys
import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailTriageEnv, TASKS, EMAILS

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-3.5-turbo")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")
env = EmailTriageEnv()

class ActionRequest(BaseModel):
    action: str

@app.post("/reset")
def reset():
    obs, info = env.reset()
    return {"observation": obs, "info": info}

@app.post("/step")
def step(req: ActionRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
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

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Baseline Agent ────────────────────────────────────────────────────────────

def run_agent():
    """Run the baseline LLM agent across all 3 tasks and emit structured logs."""
    from openai import OpenAI

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
    )

    all_scores = {}

    for task_idx, task in enumerate(TASKS):
        task_scores = []

        for email_idx, email in enumerate(EMAILS[:3]):   # 3 emails per task
            env.set_task(task_idx, email_idx)
            obs, _ = env.reset()

            # ── [START] ──────────────────────────────────────────────────────
            print(json.dumps({
                "event":      "START",
                "task_id":    task["id"],
                "task_name":  task["name"],
                "difficulty": task["difficulty"],
                "email_id":   email["id"],
                "subject":    obs["email_subject"],
            }), flush=True)

            prompt = (
                f"Task: {task['description']}\n\n"
                f"Email Subject: {obs['email_subject']}\n"
                f"Email Body: {obs['email_body']}\n\n"
                f"Your response:"
            )

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a professional email triage assistant. Be concise and accurate."},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=300,
                    temperature=0.2,
                )
                action = response.choices[0].message.content.strip()
            except Exception as e:
                action = f"[ERROR: {str(e)}]"

            obs2, reward, terminated, truncated, info = env.step(action)

            # ── [STEP] ───────────────────────────────────────────────────────
            print(json.dumps({
                "event":      "STEP",
                "task_id":    task["id"],
                "step":       1,
                "action":     action[:120] + ("..." if len(action) > 120 else ""),
                "reward":     reward,
                "terminated": terminated,
                "truncated":  truncated,
                "reason":     info.get("reward_reason", ""),
            }), flush=True)

            # ── [END] ────────────────────────────────────────────────────────
            print(json.dumps({
                "event":        "END",
                "task_id":      task["id"],
                "email_id":     email["id"],
                "total_reward": reward,
                "steps":        1,
            }), flush=True)

            task_scores.append(reward)

        avg = round(sum(task_scores) / len(task_scores), 3)
        all_scores[task["id"]] = avg
        print(json.dumps({
            "event":     "TASK_SUMMARY",
            "task_id":   task["id"],
            "avg_score": avg,
            "scores":    task_scores,
        }), flush=True)

    print("\n=== BASELINE RESULTS ===")
    for tid, score in all_scores.items():
        print(f"  {tid}: {score:.3f}")
    overall = round(sum(all_scores.values()) / len(all_scores), 3)
    print(f"  OVERALL: {overall:.3f}")
    print("========================\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--agent" in sys.argv:
        run_agent()
    else:
        uvicorn.run("inference:app", host="0.0.0.0", port=7860, reload=False)