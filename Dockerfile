# Stage 1: Convert model to ONNX
FROM python:3.9-slim AS converter

WORKDIR /build

RUN pip install --no-cache-dir sentence-transformers torch onnx transformers

# Download model and export to ONNX
RUN python -c "
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch, os

model_name = 'sankalpakc/NepaliKD-SentenceTransformers-paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device='cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)

transformer = model[0].auto_model.cpu()
transformer.eval()

os.makedirs('/build/model', exist_ok=True)

dummy = tokenizer('sample', padding='max_length', truncation=True, max_length=128, return_tensors='pt')

with torch.no_grad():
    torch.onnx.export(
        transformer,
        (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids']),
        '/build/model/model.onnx',
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
        },
        opset_version=14,
    )

tokenizer.save_pretrained('/build/model')
print('ONNX export done')
"

# Stage 2: Lightweight runtime
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=converter /build/model /app/model
COPY . .

ENV MODEL_DIR=/app/model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--forwarded-allow-ips", "*"]
