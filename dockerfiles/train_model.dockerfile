# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY configs/ configs/

WORKDIR /
RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/train_model.py"]
