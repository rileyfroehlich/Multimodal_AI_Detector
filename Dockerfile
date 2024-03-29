FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y ffmpeg

ENV PYTHONUNBUFFERED True

COPY ./requirements.txt /requirements.txt

ENV APP_HOME /app
WORKDIR ${APP_HOME}
COPY ./app/ ./

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

ENV PORT=8000
EXPOSE $PORT

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT