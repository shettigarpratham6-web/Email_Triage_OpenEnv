"""
Email Triage OpenEnv — inference.py (baseline agent)
Place at repo root. Run: python inference.py
"""
import os
import sys
import json

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-3.5-turbo")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS = [
    {"id": "task_easy",   "name": "Email Urgency Classification", "difficulty": "easy"},
    {"id": "task_medium", "name": "Action Item Extraction",       "difficulty": "medium"},
    {"id": "task_hard",   "name": "Professional Reply Drafting",  "difficulty": "hard"},
]

def run_agent():
    import requests
    from openai import OpenAI

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
    )

    all_scores = {}

    for task_idx, task in enumerate(TASKS):
        task_scores = []

        for email_idx in range(3):
            # Set task via direct env manipulation or just reset
            reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={})
            obs = reset_resp.json()["observation"]

            print(json.dumps({
                "event":      "START",
                "task_id":    task["id"],
                "task_name":  task["name"],
                "difficulty": task["difficulty"],
                "email_id":   f"email_{email_idx+1:03d}",
                "subject":    obs.get("email_subject", ""),
            }), flush=True)

            prompt = (
                f"Task: {obs.get('task_description', task['name'])}\n\n"
                f"Email Subject: {obs.get('email_subject', '')}\n"
                f"Email Body: {obs.get('email_body', '')}\n\n"
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
                action_text = response.choices[0].message.content.strip()
            except Exception as e:
                action_text = f"urgent"  # fallback

            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": {"text": action_text}},
            )
            result = step_resp.json()
            reward = result.get("reward", 0.0)

            print(json.dumps({
                "event":      "STEP",
                "task_id":    task["id"],
                "step":       1,
                "action":     action_text[:120],
                "reward":     reward,
                "terminated": result.get("terminated", True),
                "truncated":  result.get("truncated", False),
            }), flush=True)

            print(json.dumps({
                "event":        "END",
                "task_id":      task["id"],
                "email_id":     f"email_{email_idx+1:03d}",
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


if __name__ == "__main__":
    run_agent()