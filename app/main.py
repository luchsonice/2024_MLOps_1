from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Root"}

@app.get("/interference/{image_path}")
def read_item(image_path: str):
    return {"image_path": image_path}