# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "ops_dashboard.py", "--server.port=8080", "--server.enableCORS=false"]