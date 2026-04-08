from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailTriageEnv, TASKS

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")
env = EmailTriageEnv()

# ✅ Required request model
class ActionRequest(BaseModel):
    action: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs, info = env.reset()
    return {
        "observation": obs,
        "info": info
    }

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