FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-alpine3.10
RUN pip install sentencepiece requests
EXPOSE 80
COPY ./app /app/app
COPY ./model /model
