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

class PredictionRequest(BaseModel):
    """
    textにstringまたはstringの配列を渡してください

    example:
    ```json
    {
        "text": "よろしくおねがいします。"
    }
    ```
    ```json
    {
        "text": ["よろしくおねがいします。", "これってこういうことですか"]
    }
    ```
    """
    text: Union[str, List[str]]

class PredictionResult(BaseModel):
    """
    BERTによる分類の推論結果
    """
    text: str
    prediction: List[float]

class PredictionResultBody(BaseModel):
    """
    PredictionRequestのtextがstringかstringの配列かに応じて、resultもPredictionResultかその配列になる.  
    PredictionResultの確率の配列に対応するラベル名がlabelsに示される

    example:
    ```json
    {
      "result": [
        {
          "text": "よろしくおねがいします。",
          "prediction": [
            0.999928951,
            0.0000227248311,
            0.0000257293232,
            0.0000224753567
          ]
        },
        {
          "text": "これってこういうことですか",
          "prediction": [
            0.000137376017,
            0.000101430735,
            0.999649286,
            0.000111882604
          ]
        }
      ],
      "labels": [
        "Report",
        "Request",
        "Question",
        "Log code"
      ]
    }
    ```
    """
    result: Union[PredictionResult, List[PredictionResult]]
    labels: List[str]

@app.post("/tokenize", response_model=PredictionResultBody)
def tokenize(req: PredictionRequest):
    """
    request bodyとしてjsonを渡してください  
    textにstringまたはstringの配列を渡してください

    example:
    ```json
    {
        "text": "よろしくおねがいします。"
    }
    ```
    ```json
    {
        "text": ["よろしくおねがいします。", "これってこういうことですか"]
    }
    ```
    """
    if(isinstance(req.text, list)):
        result = text_to_result_list(req.text)
    else:
        result = text_to_result(req.text)

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
