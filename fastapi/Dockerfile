
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
# COPY ./app /app
# COPY ./model /model
# RUN pip install sentencepiece requests
# 
FROM python:3.7

RUN pip install fastapi uvicorn
RUN pip install sentencepiece requests

EXPOSE 80

COPY ./app /app
COPY ./model /model

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
