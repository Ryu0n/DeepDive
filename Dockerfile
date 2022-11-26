FROM python:3.8

WORKDIR /app

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--port", "8000"]
