# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.14-rc
ARG OS_VERSION=bookworm
FROM python:${PYTHON_VERSION}-slim-${OS_VERSION}

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "syntheticdatagen.py", "--type", "5"]