FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "analytics_dashboard_fixed.py", "--server.port=8501", "--server.address=0.0.0.0"]