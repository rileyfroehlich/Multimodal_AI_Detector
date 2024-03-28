FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN apt-get update && \
    apt-get install -y ffmpeg

ENV PYTHONUNBUFFERED True

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./app /app

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 200 main:app