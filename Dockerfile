# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim


RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY artifacts /app/artifacts
COPY src /app/src
COPY templates /app/templates
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt



RUN python -m pip install -r requirements.txt


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
