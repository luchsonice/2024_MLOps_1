FROM python:3.10-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install -e . --no-cache-dir

WORKDIR /app

COPY app/main.py main.py

CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1
