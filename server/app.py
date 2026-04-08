from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailTriageEnv

app = FastAPI()
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

@app.get("/health")
def health():
    return {"status": "ok"}