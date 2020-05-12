FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
RUN pip install --upgrade pip && pip install sentencepiece requests
EXPOSE 80
COPY ./app /app/app
COPY ./model /model
