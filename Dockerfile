# Container for HF Spaces (Streamlit). Build: docker build -t triage .  Run: docker run -p 8501:8501 --env-file .env triage
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app.py .
# The classifier comes from the Hub (set CLASSIFIER_DIR=<user>/<repo>); the index is built on first start.
ENV CLASSIFIER_DIR=models/ticket-classifier
EXPOSE 8501
CMD ["sh", "-c", "python -m src.data && python -m src.rag build && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
