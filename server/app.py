from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {
        "observation": {"msg": "reset"},
        "info": {}
    }

@app.post("/step")
def step():
    return {
        "observation": {"msg": "step"},
        "reward": 0.5,
        "terminated": False,
        "truncated": False,
        "info": {}
    }

@app.get("/state")
def state():
    return {"state": "running"}