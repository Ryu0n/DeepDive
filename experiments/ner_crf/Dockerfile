FROM python:3.6

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080",  "--reload"]