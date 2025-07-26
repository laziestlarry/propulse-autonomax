
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY autonomax_tactical_engine.py /app/
COPY ops_dashboard.py /app/

RUN pip install --no-cache-dir streamlit praw

EXPOSE 8080
ENV PORT=8080

CMD exec streamlit run ops_dashboard.py --server.port=$PORT --server.address=0.0.0.0
