from fastapi import FastAPI
from fastapi import UploadFile, File
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from src.models.model import ResNetModel

"""
First:
    Change checkpoint file path to a valid checkpoint
To launch API run: 
    uvicorn --reload --port 8000 app.main:app

Then go to: 
    http://localhost:8000/
    http://localhost:8000/docs
    http://localhost:8000/metrics
For inference run the following command
    curl -F "data=@<path_to_image>" http://127.0.0.1:8000/inference/
for example
    curl -F "data=@data/raw/Testing/smoking/smoking_0020.jpg" http://127.0.0.1:8000/inference/
"""

app = FastAPI()

from google.cloud import storage
import pickle
import os
import io
import torch
import numpy as np
from torchvision.transforms import v2
from PIL import Image
from src.data import _DATA_MEAN, _DATA_STD

BUCKET_NAME = "final-model-bucket"
MODEL_FILE = "model_to_deploy.pickle"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)

my_model = pickle.loads(blob.download_as_string())
my_model.eval()

transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(_DATA_MEAN, _DATA_STD)
])

@app.get("/")
def read_root():
    return {"Go to /metrics for metrics","Go to /inference/image_path for inference"}

@app.post("/inference/")
async def inference(data: UploadFile = File(...)):

    # Save the image
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()

    # Open the image
    img = Image.open("image.jpg")    

    # Preprocess the image
    img = torch.unsqueeze(transform(img), 0)

    # Do prediction
    with torch.no_grad():
        y_hat = my_model(img)

    # Get class
    prediction = np.argmax(y_hat.numpy())
    predicted_class = ["notsmoking", "smoking"][prediction]

    return {"predicted_class": predicted_class, "raw_logits": str(y_hat.numpy()[0])}

Instrumentator().instrument(app).expose(app)