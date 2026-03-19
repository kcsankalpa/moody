FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sankalpakc/NepaliKD-SentenceTransformers-paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--forwarded-allow-ips", "*"]
