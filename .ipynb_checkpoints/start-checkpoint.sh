#!/bin/bash
export HF_HOME="/hpcwork/yw701564/huggingface_cache"
export TRANSFORMERS_CACHE="/hpcwork/yw701564/huggingface_cache"

uvicorn api:app --host 0.0.0.0 --port 8000
# ./ngrok http 8000