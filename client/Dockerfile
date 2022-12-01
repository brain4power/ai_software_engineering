FROM python:3.10.6

WORKDIR /code

RUN apt-get update && \
    apt-get install libsndfile1-dev ffmpeg -y

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/client

RUN ["chmod", "+x", "/code/client/run_client.sh"]
