FROM python:3.9-slim

COPY ./requirements.txt /SnakeAI/requirements.txt
RUN pip install --no-cache-dir -r /SnakeAI/requirements.txt

COPY . /SnakeAI

EXPOSE 80

CMD ["python", "main.py"]