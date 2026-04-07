from fastapi import FastAPI
from pydantic import BaseModel
from env import MazeEnv

app = FastAPI()

env = MazeEnv()

# Request body format
class Action(BaseModel):
    action: int

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": list(state)}

@app.post("/step")
def step(action: Action):
    state, reward, done = env.step(action.action)
    return {
        "state": list(state),
        "reward": reward,
        "done": done
    }