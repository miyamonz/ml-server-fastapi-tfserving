from fastapi import FastAPI
app = FastAPI()

import sentencepiece as sp
vocab_file = "/model/wiki-ja.vocab"
model_file = "/model/wiki-ja.model"
tokenizer = sp.SentencePieceProcessor()
tokenizer.Load(model_file)

from . import req

@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.get("/tokenize/{text}")
def tokenize(text: str):
    instance = req.create_instance(text)
    prediction = req.send_request([instance])

    return {
            "prediction": prediction,
            }
