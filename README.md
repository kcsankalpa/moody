Moody is an FastAPI service that is used to ingest Nepali song lyrics and fetch them using an embedding model <br/> <a href="https://huggingface.co/sankalpakc/NepaliKD-MiniLM-L12-v2-onnx"><strong>Link </strong> </a> 
and qdrant vector database.

The embedding model is used is:
MiniLM-L12-v2 --> (Knowledge Distilled) NepaliKD-MiniLM-L12-v2-onnx --> (converted to onnx model) NepaliKD-MiniLM-L12-v2-onnx

Further more, this is a lightweight implemenation (pytorch is not used and Tokenizer is used) so that the model size is dramatically reduced to ~100Mb and can be deployed as a usable service effectively. 

It is intended to be conusmed using MCP servers for searching Nepali songs according to ones mood and optionally an artist's name.
