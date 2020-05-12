#latestは再ビルド時に意図せず変化が起きる可能性があるので本番時には固定しよう
FROM tensorflow/serving:latest
ARG MY_MODEL_NAME="my_model_name"
ENV MODEL_NAME=$MY_MODEL_NAME
ARG LOCAL_MODEL_PATH="./export"
COPY $LOCAL_MODEL_PATH /models/${MY_MODEL_NAME}
