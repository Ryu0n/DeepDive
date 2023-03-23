FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app/

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . /app/

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port 8080 --reload