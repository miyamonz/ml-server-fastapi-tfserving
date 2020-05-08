from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

from . import req

@app.get("/")
def read_root():
    return {"Hello": "World!"}

class Item(BaseModel):
    text: str

@app.post("/tokenize")
def tokenize(item: Item):
    instance = req.create_instance(item.text)
    prediction = req.send_request([instance])

    return {
            "text": item.text,
            "prediction": prediction,
            }
