from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"message": "reset ok"}

@app.post("/step")
def step():
    return {"message": "step ok"}
