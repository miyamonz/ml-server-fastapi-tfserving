#latestは危険
FROM tensorflow/serving:latest
ARG MY_MODEL_NAME="classify_chat_text"
ENV MODEL_NAME=$MY_MODEL_NAME
ARG LOCAL_MODEL_PATH="./export"
COPY $LOCAL_MODEL_PATH /models/${MY_MODEL_NAME}
