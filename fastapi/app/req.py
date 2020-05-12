import os
import json
import requests
import sentencepiece as sp


vocab_file = "/model/wiki-ja.vocab"
model_file = "/model/wiki-ja.model"
tokenizer = sp.SentencePieceProcessor()
tokenizer.Load(model_file)

def create_instance(example):
    raw_pieces  = tokenizer.EncodeAsPieces(example)

    pieces = []
    segment_ids = []

    # first token must be CLS
    pieces.append("[CLS]")
    segment_ids.append(0)

    for piece in raw_pieces:
        pieces.append(piece)
        segment_ids.append(0)

    # last token must be SEP
    pieces.append('[SEP]')
    segment_ids.append(0)

    # convert pieces to ids
    input_ids = [ tokenizer.PieceToId(p) for p in pieces ]
    input_mask = [1] * len(input_ids)

    #fill 0 in the rest list space
    max_seq_length = 512
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_id = 0
    instances = {
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "label_ids":label_id,
            }
    return instances

MY_MODEL_NAME = os.environ["MY_MODEL_NAME"]
endpoints = f"http://servingtf:8501/v1/models/{MY_MODEL_NAME}:predict"
headers = {"content-type":"application-json"}
def send_request(instances):
    data = json.dumps({"instances":instances})
    response = requests.post(endpoints, data=data, headers=headers)
    prediction = json.loads(response.text)['predictions']
    return prediction
