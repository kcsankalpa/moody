FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download quantized ONNX model and tokenizer from HuggingFace
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sankalpakc/NepaliKD-MiniLM-L12-v2-onnx', local_dir='/app/model')"

COPY . .

ENV MODEL_DIR=/app/model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--forwarded-allow-ips", "*"]
