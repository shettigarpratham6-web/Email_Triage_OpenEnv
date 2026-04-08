from fastapi import FastAPI
from pydantic import BaseModel
from env import MazeEnv

app = FastAPI()

env = MazeEnv()

class Action(BaseModel):
    action: int

@app.post("/reset")
def reset():
    result = env.reset()
    
    # Handle both (obs, info) tuple and plain obs
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs, info = result, {}

    return {
    "observation": obs.tolist(),  # ← was list(obs)
    "info": info
}

@app.post("/step")
def step(action: Action):
    result = env.step(action.action)

    # Handle both 5-value and 3-value returns
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done = result
        terminated = done
        truncated = False
        info = {}

    return {
    "observation": obs.tolist(),  # ← was list(obs)
    "reward": float(reward),
    "terminated": bool(terminated),   # ← add bool()
    "truncated": bool(truncated),     # ← add bool()
    "info": info
}