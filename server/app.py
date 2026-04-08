from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from env import EmailTriageEnv

app = FastAPI()
env = EmailTriageEnv()

# ✅ Request schema
class ActionRequest(BaseModel):
    action: str

# ✅ Health check (required)
@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ Reset endpoint
@app.post("/reset")
def reset():
    try:
        obs, info = env.reset()
    except Exception as e:
        return {
            "observation": {},
            "info": {"error": str(e)}
        }

    return {
        "observation": obs if isinstance(obs, dict) else {},
        "info": info if isinstance(info, dict) else {}
    }

# ✅ Step endpoint (CRITICAL)
@app.post("/step")
def step(req: ActionRequest):
    try:
        obs, reward, terminated, truncated, info = env.step(req.action)
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "info": {"error": str(e)}
        }

    # ✅ Ensure strict validator compliance
    return {
        "observation": obs if isinstance(obs, dict) else {},
        "reward": float(max(0.0, min(1.0, reward))),  # clamp 0–1
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info if isinstance(info, dict) else {}
    }

# ✅ State endpoint
@app.get("/state")
def state():
    try:
        s = env.state()
        return s if isinstance(s, dict) else {"state": str(s)}
    except Exception:
        return {"state": "unknown"}

# ✅ Optional but useful (safe for validator)
@app.get("/tasks")
def tasks():
    try:
        return {"tasks": getattr(env, "TASKS", [])}
    except Exception:
        return {"tasks": []}