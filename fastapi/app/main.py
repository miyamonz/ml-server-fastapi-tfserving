from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

from . import req

# ラベルデータはtensorflowのembedding_lookupなどを使えば良いかもしれないが、tensorflowに慣れてないので普通にファイルで渡す
import os
APP_DIR = os.path.abspath(os.path.dirname(__file__))
labels = []
with open(APP_DIR + '/labels.txt') as f:
    labels = f.read().splitlines()


@app.get("/")
def read_root():
    return {"Hello": "World!"}

class Item(BaseModel):
    text: Union[str, List[str]]

@app.post("/tokenize")
def tokenize(item: Item):
    if(isinstance(item.text, list)):
        result = text_to_result_list(item.text)
    else:
        result = text_to_result(item.text)

    return {
            "result": result,
            "labels": labels,
            }

def text_to_result(text: str):
    instances = [req.create_instance(text)]
    prediction = req.send_request(instances)
    return {
            "text": text,
            "prediction": prediction[0],
            }

# tensorflowに投げるときにまとめて投げたほうが速いのでmap的なことをせずに別関数として定義すつ
def text_to_result_list(text: List[str]):
    instances = [req.create_instance(t) for t in text] 
    prediction = req.send_request(instances)
    zipped = zip(text, prediction)
    return [ {"text": t, "prediction": p } for (t, p) in zipped ]
